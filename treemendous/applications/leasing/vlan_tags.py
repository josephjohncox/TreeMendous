"""Network-scoped VLAN tag leases constrained to IEEE tag values."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from treemendous.applications._shared.clock import Clock
from treemendous.applications._shared.leasing import LeaseUnavailableError
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
    spans_without,
)
from treemendous.domain import Span, validate_coordinate

VlanLease = NumericLease


class VlanScopeError(ValueError):
    """Raised when a VLAN request names an unknown network scope."""


class VlanUnavailableError(LeaseUnavailableError):
    """Raised when no requested allowed VLAN tag block is available."""


@dataclass(frozen=True)
class VlanPoolCheckpoint:
    """Restorable network-scoped VLAN pool lineages."""

    group: GroupCheckpoint


class VlanTagPool:
    """Lease contiguous VLAN IDs in 1..4094 for each network scope.

    VLAN 0 and 4095 never enter the allocation domain. Additional global and
    per-network inclusive ranges are excluded before LeasePool construction.
    The same numeric VLAN can be independently leased in different scopes.
    """

    def __init__(
        self,
        scopes: Iterable[str] = ("default",),
        *,
        reserved_ranges: Iterable[tuple[int, int]] = (),
        scope_reserved: Mapping[str, Iterable[tuple[int, int]]] | None = None,
        clock: Clock | None = None,
    ) -> None:
        normalized_scopes = tuple(
            require_string(scope, "network scope") for scope in scopes
        )
        if not normalized_scopes:
            raise ValueError("at least one network scope is required")
        if len(set(normalized_scopes)) != len(normalized_scopes):
            raise ValueError("network scopes must be unique")
        globally_reserved = tuple(
            inclusive_span(start, end, "reserved VLAN range")
            for start, end in reserved_ranges
        )
        scoped = {} if scope_reserved is None else dict(scope_reserved)
        unknown = set(scoped).difference(normalized_scopes)
        if unknown:
            raise VlanScopeError(
                f"reserved policy names unknown scopes: {sorted(unknown)!r}"
            )
        domain = inclusive_span(1, 4094, "VLAN domain")
        domains: dict[str, tuple[Span, ...]] = {}
        for scope in normalized_scopes:
            local = tuple(
                inclusive_span(start, end, "scope-reserved VLAN range")
                for start, end in scoped.get(scope, ())
            )
            domains[scope] = spans_without(domain, (*globally_reserved, *local))
        self._group = PoolGroup(
            domains,
            clock=clock if clock is not None else ProcessClock(),
        )

    @classmethod
    def from_checkpoint(
        cls, checkpoint: VlanPoolCheckpoint, *, clock: Clock
    ) -> VlanTagPool:
        """Restore into fresh process-local scope lineages."""
        if not isinstance(checkpoint, VlanPoolCheckpoint):
            raise TypeError("checkpoint must be a VlanPoolCheckpoint")
        engine = cls.__new__(cls)
        engine._group = PoolGroup.restore(checkpoint.group, clock=clock)
        domain = inclusive_span(1, 4094, "VLAN domain")
        if any(
            not domain.contains(span)
            for pool in engine._group.pools.values()
            for span in pool.allowed_spans
        ):
            raise ValueError("VLAN checkpoint domain extends outside 1..4094")
        return engine

    def acquire(
        self,
        scope: str,
        owner: str,
        *,
        ttl: int,
        count: int = 1,
        start_tag: int | None = None,
        request_id: str | None = None,
    ) -> VlanLease:
        """Acquire the earliest VLAN block or an exact allowed block."""
        if scope not in self._group.pools:
            raise VlanScopeError(f"unknown network scope: {scope!r}")
        count = require_positive(count, "count")
        exact = None
        if start_tag is not None:
            validate_coordinate(start_tag, "start_tag")
            if start_tag < 1 or start_tag + count > 4095:
                raise ValueError("VLAN block must stay inside 1..4094")
            exact = Span(start_tag, start_tag + count)
        try:
            return self._group.acquire(
                scope,
                owner,
                ttl=ttl,
                size=count,
                exact_span=exact,
                request_id=request_id,
            )
        except LeaseUnavailableError as exc:
            raise VlanUnavailableError(str(exc)) from None

    def renew(self, handle: VlanLease, *, ttl: int) -> VlanLease:
        """Renew a current VLAN lease."""
        return self._group.renew(handle, ttl=ttl)

    def release(self, handle: VlanLease) -> VlanLease:
        """Release a current VLAN lease."""
        return self._group.release(handle)

    def expire(self) -> tuple[VlanLease, ...]:
        """Materialize elapsed VLAN leases."""
        return self._group.expire()

    def validate_fence(self, handle: VlanLease, tag: int) -> bool:
        """Fence one VLAN under a stable network-scope/tag key."""
        validate_coordinate(tag, "tag")
        if tag < handle.resource.start or tag >= handle.resource.end:
            raise ValueError("tag is outside the leased block")
        key = ("vlan-tag-pools", handle.scope, tag)
        return self._group.validate_fence(key, handle)

    def snapshot(self) -> GroupSnapshot:
        """Return immutable VLAN pool snapshots by network scope."""
        return self._group.snapshot()

    def checkpoint(self) -> VlanPoolCheckpoint:
        """Return pool state without process-local downstream fence marks."""
        return VlanPoolCheckpoint(self._group.checkpoint())

    def diagnostics(self) -> GroupDiagnostics:
        """Return lifecycle and capacity counters by network scope."""
        return self._group.diagnostics()


def create_engine(**kwargs: Any) -> VlanTagPool:
    """Create the manifest factory for VLAN tag pools."""
    return VlanTagPool(**kwargs)
