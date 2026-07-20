"""Shard-scoped game region leasing, adjacency, and fenced handoff."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from treemendous.applications._shared.clock import Clock
from treemendous.applications._shared.leasing import (
    ForeignLeaseError,
    LeaseState,
    LeaseUnavailableError,
    StaleLeaseError,
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
class _HandoffRecord:
    request_id: str
    shard: str
    source_token: int
    new_owner: str
    ttl: int
    result_token: int


@dataclass(frozen=True)
class RegionPoolCheckpoint:
    """Restorable shard pools and idempotent ownership handoffs."""

    shards: tuple[tuple[str, tuple[int, int]], ...]
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
        engine.shards = dict(checkpoint.shards)
        engine._handoffs = {item.request_id: item for item in checkpoint.handoffs}
        if len(engine._handoffs) != len(checkpoint.handoffs):
            raise ValueError("handoff request IDs must be unique")
        engine._group = PoolGroup.restore(checkpoint.group, clock=clock)
        if set(engine.shards) != set(engine._group.pools):
            raise ValueError("region checkpoint shards do not match its pools")
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
        if shard not in self.shards:
            raise ValueError(f"unknown shard: {shard!r}")
        count = require_positive(count, "count")
        if start_region is not None and adjacent_to is not None:
            raise ValueError("start_region and adjacent_to are mutually exclusive")
        with self._group.lock:
            exact = None
            if start_region is not None:
                validate_coordinate(start_region, "start_region")
                exact = Span(start_region, start_region + count)
            elif adjacent_to is not None:
                anchor = self._active_anchor(shard, owner, adjacent_to)
                snapshot = self._group.pool(shard).snapshot()
                prior = next(
                    (
                        lease
                        for lease in snapshot.leases
                        if request_id is not None
                        and lease.request_id == request_id
                    ),
                    None,
                )
                if prior is not None:
                    exact = prior.resource
                else:
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
                return self._group.acquire(
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
        with self._group.lock:
            pool = self._group.pool(handle.scope)
            if handle.pool_id != pool.pool_id:
                raise ForeignLeaseError("region lease belongs to another lineage")
            issued = next(
                (
                    lease
                    for lease in pool.snapshot().leases
                    if lease.token == handle.token
                ),
                None,
            )
            if (
                issued is None
                or issued.owner != handle.owner
                or issued.resource != handle.resource
                or issued.acquired_at != handle.lease.acquired_at
            ):
                raise StaleLeaseError("region handoff evidence is not issued history")
            existing = self._handoffs.get(request_id)
            fingerprint = (handle.scope, handle.token, new_owner, ttl)
            if existing is not None:
                recorded = (
                    existing.shard,
                    existing.source_token,
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
                    raise RegionHandoffError("handoff result is absent from pool history")
                return NumericLease(existing.shard, result)
            self._group.release(handle)
            try:
                transferred = self._group.acquire(
                    handle.scope,
                    new_owner,
                    ttl=ttl,
                    size=handle.resource.length,
                    exact_span=handle.resource,
                    request_id=None,
                )
            except LeaseUnavailableError as exc:
                raise RegionHandoffError(str(exc)) from None
            self._handoffs[request_id] = _HandoffRecord(
                request_id,
                handle.scope,
                handle.token,
                new_owner,
                ttl,
                transferred.token,
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
        return RegionPoolCheckpoint(
            tuple(sorted(self.shards.items())),
            tuple(self._handoffs[key] for key in sorted(self._handoffs)),
            self._group.checkpoint(),
        )

    def diagnostics(self) -> GroupDiagnostics:
        """Return lifecycle and capacity counters by shard."""
        return self._group.diagnostics()


def create_engine(**kwargs: Any) -> GameRegionPool:
    """Create the manifest factory for game world region IDs."""
    return GameRegionPool(**kwargs)
