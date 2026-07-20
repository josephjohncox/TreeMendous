"""Shard-scoped game region leasing, adjacency, and fenced handoff."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from treemendous.applications._shared.clock import Clock
from treemendous.applications._shared.leasing import (
    LeaseRequestConflictError,
    LeaseState,
    LeaseUnavailableError,
)
from treemendous.applications.leasing._common import (
    GroupCheckpoint,
    GroupDiagnostics,
    GroupSnapshot,
    NumericLease,
    PoolGroup,
    ProcessClock,
    inclusive_span,
    require_positive,
    require_string,
)
from treemendous.domain import Span, validate_coordinate

RegionLease = NumericLease


class RegionUnavailableError(LeaseUnavailableError):
    """Raised when no requested region band is available on a shard."""


class RegionAdjacencyError(ValueError):
    """Raised when an adjacency anchor is foreign, stale, or differently owned."""


class RegionHandoffError(RuntimeError):
    """Raised when handoff evidence conflicts with a prior transfer."""


@dataclass(frozen=True)
class _AnchorIdentity:
    shard: str
    pool_id: str
    owner: str
    token: int
    resource: Span
    acquired_at: int


@dataclass(frozen=True)
class _RegionRequest:
    request_id: str
    shard: str
    owner: str
    ttl: int
    count: int
    selection_mode: str
    start_region: int | None
    anchor: _AnchorIdentity | None
    result_token: int
    result_resource: Span


@dataclass(frozen=True)
class _LeaseIdentity:
    pool_id: str
    token: int
    owner: str
    resource: Span
    acquired_at: int
    request_id: str | None


@dataclass(frozen=True)
class _HandoffRecord:
    request_id: str
    shard: str
    source_token: int
    source_owner: str
    source_resource: Span
    source_acquired_at: int
    source_expires_at: int
    source_revision: int
    source_request_id: str | None
    new_owner: str
    ttl: int
    result_token: int
    result_identity: _LeaseIdentity


@dataclass(frozen=True)
class RegionPoolCheckpoint:
    """Restorable shard pools and idempotent ownership operations."""

    shards: tuple[tuple[str, tuple[int, int]], ...]
    requests: tuple[_RegionRequest, ...]
    handoffs: tuple[_HandoffRecord, ...]
    group: GroupCheckpoint


class GameRegionPool:
    """Lease contiguous region IDs within independent shard namespaces.

    Optional adjacency anchors must be active leases on the same shard and
    owned by the same server. Handoff releases an old owner and reacquires the
    exact band for a new owner, yielding a higher fencing token. The transition
    is serialized only in this process; durable cross-process ownership and
    atomic downstream token enforcement remain external responsibilities.
    """

    def __init__(
        self,
        shards: Mapping[str, tuple[int, int]] | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        configured = {"default": (1, 1024)} if shards is None else dict(shards)
        if not configured:
            raise ValueError("at least one shard is required")
        self.shards: dict[str, tuple[int, int]] = {}
        domains: dict[str, tuple[Span, ...]] = {}
        for shard, bounds in configured.items():
            shard = require_string(shard, "shard")
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise TypeError("shard bounds must be inclusive (first, last) tuples")
            span = inclusive_span(bounds[0], bounds[1], "shard region range")
            self.shards[shard] = bounds
            domains[shard] = (span,)
        self._requests: dict[str, _RegionRequest] = {}
        self._handoffs: dict[str, _HandoffRecord] = {}
        self._group = PoolGroup(
            domains,
            clock=clock if clock is not None else ProcessClock(),
        )

    @classmethod
    def from_checkpoint(
        cls, checkpoint: RegionPoolCheckpoint, *, clock: Clock
    ) -> GameRegionPool:
        """Restore into fresh shard lineages after external writer takeover."""
        if not isinstance(checkpoint, RegionPoolCheckpoint):
            raise TypeError("checkpoint must be a RegionPoolCheckpoint")
        engine = cls.__new__(cls)
        engine.shards = {}
        domains: dict[str, tuple[Span, ...]] = {}
        for shard, bounds in checkpoint.shards:
            shard = require_string(shard, "shard")
            if shard in engine.shards:
                raise ValueError("checkpoint shard names must be unique")
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise TypeError("shard bounds must be inclusive tuples")
            domain = inclusive_span(bounds[0], bounds[1], "shard region range")
            engine.shards[shard] = bounds
            domains[shard] = (domain,)
        if not engine.shards:
            raise ValueError("checkpoint must contain at least one shard")
        engine._group = PoolGroup.restore(checkpoint.group, clock=clock)
        engine._group.require_domains(domains)

        engine._requests = {}
        source_lineages = {
            entry.scope: entry.source_pool_id for entry in checkpoint.group.pools
        }
        source_history = {
            (entry.scope, lease.token): lease
            for entry in checkpoint.group.pools
            for lease in entry.pool.leases
        }
        history = {
            (scope, lease.token): lease
            for scope, snapshot in engine._group.snapshot().pools
            for lease in snapshot.leases
        }
        for record in checkpoint.requests:
            if not isinstance(record, _RegionRequest):
                raise TypeError("region checkpoint contains an invalid request")
            require_string(record.request_id, "request_id")
            require_string(record.owner, "request owner")
            require_positive(record.ttl, "request ttl")
            require_positive(record.count, "request count")
            if record.request_id in engine._requests:
                raise ValueError("region request IDs must be unique")
            if record.shard not in engine.shards:
                raise ValueError("region request names an unknown shard")
            if record.selection_mode not in {"automatic", "exact", "adjacent"}:
                raise ValueError("region request has an invalid selection mode")
            if (record.selection_mode == "exact") != (record.start_region is not None):
                raise ValueError("region request start conflicts with selection mode")
            if record.start_region is not None:
                validate_coordinate(record.start_region, "request start_region")
                if record.result_resource.start != record.start_region:
                    raise ValueError("region request start conflicts with its result")
            if (record.selection_mode == "adjacent") != (record.anchor is not None):
                raise ValueError("region request anchor conflicts with selection mode")
            if record.result_resource.length != record.count:
                raise ValueError("region request count conflicts with its result")
            if record.anchor is not None:
                if not isinstance(record.anchor, _AnchorIdentity):
                    raise TypeError("region request contains an invalid anchor")
                anchor = source_history.get((record.anchor.shard, record.anchor.token))
                if (
                    anchor is None
                    or record.anchor.shard != record.shard
                    or record.anchor.pool_id != source_lineages.get(record.shard)
                    or anchor.pool_id != record.anchor.pool_id
                    or anchor.owner != record.anchor.owner
                    or anchor.resource != record.anchor.resource
                    or anchor.acquired_at != record.anchor.acquired_at
                ):
                    raise ValueError("region request anchor conflicts with history")
            lease = history.get((record.shard, record.result_token))
            if (
                lease is None
                or lease.request_id != record.request_id
                or lease.owner != record.owner
                or lease.resource != record.result_resource
            ):
                raise ValueError("region request conflicts with lease history")
            engine._requests[record.request_id] = record
        history_request_sequence = tuple(
            lease.request_id
            for lease in history.values()
            if lease.request_id is not None
        )
        history_request_ids = set(history_request_sequence)
        if len(history_request_ids) != len(history_request_sequence):
            raise ValueError("region request IDs must be unique across shards")
        if set(engine._requests) != history_request_ids:
            raise ValueError("region requests must agree with lease history")

        engine._handoffs = {}
        for item in checkpoint.handoffs:
            if not isinstance(item, _HandoffRecord):
                raise TypeError("region checkpoint contains an invalid handoff")
            require_string(item.request_id, "handoff request_id")
            require_string(item.new_owner, "handoff new_owner")
            require_positive(item.ttl, "handoff ttl")
            if item.request_id in engine._handoffs:
                raise ValueError("handoff request IDs must be unique")
            if not isinstance(item.result_identity, _LeaseIdentity):
                raise TypeError("handoff record contains an invalid result identity")
            require_string(
                item.result_identity.pool_id,
                "handoff result pool_id",
            )
            source = source_history.get((item.shard, item.source_token))
            result = source_history.get((item.shard, item.result_token))
            if (
                source is None
                or source.state is not LeaseState.RELEASED
                or source.owner != item.source_owner
                or source.resource != item.source_resource
                or source.acquired_at != item.source_acquired_at
                or source.expires_at != item.source_expires_at
                or source.revision != item.source_revision
                or source.request_id != item.source_request_id
            ):
                raise ValueError("handoff record conflicts with lease history")
            if result is None:
                raise ValueError("handoff record conflicts with lease history")
            identity = item.result_identity
            if (
                identity.pool_id != source_lineages.get(item.shard)
                or result.pool_id != identity.pool_id
                or result.token != item.result_token
                or result.token != identity.token
                or result.owner != identity.owner
                or result.resource != identity.resource
                or result.acquired_at != identity.acquired_at
                or result.request_id != identity.request_id
                or identity.owner != item.new_owner
                or identity.resource != source.resource
                or result.token <= source.token
            ):
                raise ValueError(
                    "handoff record has conflicting handoff result identity"
                )
            restored_result = history[(item.shard, item.result_token)]
            engine._handoffs[item.request_id] = replace(
                item,
                result_identity=replace(
                    identity,
                    pool_id=restored_result.pool_id,
                ),
            )
        return engine

    def _active_anchor(
        self, shard: str, owner: str, anchor: RegionLease
    ) -> RegionLease:
        if not isinstance(anchor, NumericLease):
            raise TypeError("adjacent_to must be a RegionLease")
        if anchor.scope != shard or anchor.owner != owner:
            raise RegionAdjacencyError(
                "adjacency anchor must have the same shard and owner"
            )
        current = next(
            (
                lease
                for lease in self._group.pool(shard).snapshot().leases
                if lease.token == anchor.token
            ),
            None,
        )
        if current != anchor.lease or current.state is not LeaseState.ACTIVE:
            raise RegionAdjacencyError("adjacency anchor is not active and current")
        return anchor

    def acquire(
        self,
        shard: str,
        owner: str,
        *,
        ttl: int,
        count: int = 1,
        start_region: int | None = None,
        adjacent_to: RegionLease | None = None,
        request_id: str | None = None,
    ) -> RegionLease:
        """Acquire a region band, optionally immediately beside an owned band."""
        shard = require_string(shard, "shard")
        if shard not in self.shards:
            raise ValueError(f"unknown shard: {shard!r}")
        owner = require_string(owner, "owner")
        ttl = require_positive(ttl, "ttl")
        count = require_positive(count, "count")
        if request_id is not None:
            request_id = require_string(request_id, "request_id")
        if start_region is not None and adjacent_to is not None:
            raise ValueError("start_region and adjacent_to are mutually exclusive")
        if start_region is not None:
            validate_coordinate(start_region, "start_region")
        anchor_identity = None
        if adjacent_to is not None:
            if not isinstance(adjacent_to, NumericLease):
                raise TypeError("adjacent_to must be a RegionLease")
            anchor_identity = _AnchorIdentity(
                adjacent_to.scope,
                adjacent_to.pool_id,
                adjacent_to.owner,
                adjacent_to.token,
                adjacent_to.resource,
                adjacent_to.lease.acquired_at,
            )
        selection_mode = (
            "adjacent"
            if adjacent_to is not None
            else "exact"
            if start_region is not None
            else "automatic"
        )
        fingerprint = (
            shard,
            owner,
            ttl,
            count,
            selection_mode,
            start_region,
            anchor_identity,
        )
        with self._group.lock:
            existing = None if request_id is None else self._requests.get(request_id)
            if existing is not None:
                recorded = (
                    existing.shard,
                    existing.owner,
                    existing.ttl,
                    existing.count,
                    existing.selection_mode,
                    existing.start_region,
                    existing.anchor,
                )
                if fingerprint != recorded:
                    raise LeaseRequestConflictError(
                        f"request_id {request_id!r} was already used differently"
                    )
                if existing.selection_mode == "adjacent":
                    replay_exact = existing.result_resource
                elif existing.selection_mode == "automatic":
                    replay_exact = None
                else:
                    if existing.start_region is None:
                        raise RuntimeError("exact region request lost its start")
                    replay_exact = Span(
                        existing.start_region,
                        existing.start_region + count,
                    )
                return self._group.acquire(
                    shard,
                    owner,
                    ttl=ttl,
                    size=count,
                    exact_span=replay_exact,
                    request_id=request_id,
                )

            exact = None
            if start_region is not None:
                exact = Span(start_region, start_region + count)
            elif adjacent_to is not None:
                anchor = self._active_anchor(shard, owner, adjacent_to)
                snapshot = self._group.pool(shard).snapshot()
                first, last = self.shards[shard]
                candidates = (
                    Span(anchor.resource.end, anchor.resource.end + count),
                    Span(anchor.resource.start - count, anchor.resource.start),
                )
                exact = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.start >= first
                        and candidate.end <= last + 1
                        and any(
                            available.start <= candidate.start
                            and candidate.end <= available.end
                            for available in snapshot.available_spans
                        )
                    ),
                    None,
                )
                if exact is None:
                    raise RegionAdjacencyError(
                        "no available in-shard adjacent band can fit"
                    )
            try:
                result = self._group.acquire(
                    shard,
                    owner,
                    ttl=ttl,
                    size=count,
                    exact_span=exact,
                    request_id=request_id,
                )
            except LeaseUnavailableError as exc:
                if adjacent_to is not None:
                    raise RegionAdjacencyError(
                        "adjacent region band is unavailable"
                    ) from None
                raise RegionUnavailableError(str(exc)) from None
            if request_id is not None:
                self._requests[request_id] = _RegionRequest(
                    request_id,
                    shard,
                    owner,
                    ttl,
                    count,
                    selection_mode,
                    start_region,
                    anchor_identity,
                    result.token,
                    result.resource,
                )
            return result

    def handoff(
        self,
        handle: RegionLease,
        new_owner: str,
        *,
        ttl: int,
        request_id: str,
    ) -> RegionLease:
        """Transfer an exact region band and issue a higher fencing token."""
        new_owner = require_string(new_owner, "new_owner")
        request_id = require_string(request_id, "request_id")
        ttl = require_positive(ttl, "ttl")
        if not isinstance(handle, NumericLease):
            raise TypeError("handle must be a RegionLease")
        with self._group.lock:
            existing = self._handoffs.get(request_id)
            fingerprint = (
                handle.scope,
                handle.token,
                handle.owner,
                handle.resource,
                handle.lease.acquired_at,
                handle.expires_at,
                handle.revision,
                handle.lease.request_id,
                new_owner,
                ttl,
            )
            if existing is not None:
                recorded = (
                    existing.shard,
                    existing.source_token,
                    existing.source_owner,
                    existing.source_resource,
                    existing.source_acquired_at,
                    existing.source_expires_at,
                    existing.source_revision,
                    existing.source_request_id,
                    existing.new_owner,
                    existing.ttl,
                )
                if fingerprint != recorded:
                    raise RegionHandoffError(
                        "handoff request ID was reused with different arguments"
                    )
                result = next(
                    (
                        lease
                        for lease in self._group.pool(existing.shard).snapshot().leases
                        if lease.token == existing.result_token
                    ),
                    None,
                )
                if result is None:
                    raise RegionHandoffError(
                        "handoff result is absent from pool history"
                    )
                identity = existing.result_identity
                if (
                    result.pool_id != identity.pool_id
                    or result.token != existing.result_token
                    or result.token != identity.token
                    or result.owner != identity.owner
                    or result.resource != identity.resource
                    or result.acquired_at != identity.acquired_at
                    or result.request_id != identity.request_id
                ):
                    raise RegionHandoffError(
                        "handoff result identity differs from pool history"
                    )
                return NumericLease(existing.shard, result)
            try:
                transferred = self._group.transfer(handle, new_owner, ttl=ttl)
            except LeaseUnavailableError as exc:
                raise RegionHandoffError(str(exc)) from None
            self._handoffs[request_id] = _HandoffRecord(
                request_id,
                handle.scope,
                handle.token,
                handle.owner,
                handle.resource,
                handle.lease.acquired_at,
                handle.expires_at,
                handle.revision,
                handle.lease.request_id,
                new_owner,
                ttl,
                transferred.token,
                _LeaseIdentity(
                    transferred.pool_id,
                    transferred.token,
                    transferred.owner,
                    transferred.resource,
                    transferred.lease.acquired_at,
                    transferred.lease.request_id,
                ),
            )
            return transferred

    def renew(self, handle: RegionLease, *, ttl: int) -> RegionLease:
        """Renew a current region owner lease."""
        return self._group.renew(handle, ttl=ttl)

    def release(self, handle: RegionLease) -> RegionLease:
        """Release a current region band."""
        return self._group.release(handle)

    def expire(self) -> tuple[RegionLease, ...]:
        """Materialize elapsed region ownership."""
        return self._group.expire()

    def validate_fence(self, handle: RegionLease, region_id: int) -> bool:
        """Fence one region using its stable shard/region key."""
        validate_coordinate(region_id, "region_id")
        if region_id < handle.resource.start or region_id >= handle.resource.end:
            raise ValueError("region_id is outside the leased band")
        key = ("game-world-region-ids", handle.scope, region_id)
        return self._group.validate_fence(key, handle)

    def snapshot(self) -> GroupSnapshot:
        """Return immutable shard ownership snapshots."""
        return self._group.snapshot()

    def checkpoint(self) -> RegionPoolCheckpoint:
        """Return shard/handoff state without downstream fence high-water marks."""
        with self._group.lock:
            group = self._group.checkpoint()
            lineages = {entry.scope: entry.source_pool_id for entry in group.pools}
            requests = tuple(
                replace(
                    record,
                    anchor=(
                        None
                        if record.anchor is None
                        else replace(
                            record.anchor,
                            pool_id=lineages[record.anchor.shard],
                        )
                    ),
                )
                for record in (self._requests[key] for key in sorted(self._requests))
            )
            handoffs = tuple(
                replace(
                    record,
                    result_identity=replace(
                        record.result_identity,
                        pool_id=lineages[record.shard],
                    ),
                )
                for record in (self._handoffs[key] for key in sorted(self._handoffs))
            )
            return RegionPoolCheckpoint(
                tuple(sorted(self.shards.items())),
                requests,
                handoffs,
                group,
            )

    def diagnostics(self) -> GroupDiagnostics:
        """Return lifecycle and capacity counters by shard."""
        return self._group.diagnostics()


def create_engine(**kwargs: Any) -> GameRegionPool:
    """Create the manifest factory for game world region IDs."""
    return GameRegionPool(**kwargs)
