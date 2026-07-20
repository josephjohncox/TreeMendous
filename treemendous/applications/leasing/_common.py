"""Private helpers for the process-local numeric leasing applications."""

from __future__ import annotations

import time
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from threading import RLock

from treemendous.applications._shared.clock import Clock
from treemendous.applications._shared.leasing import (
    FenceValidator,
    ForeignLeaseError,
    InvalidLeaseError,
    Lease,
    LeaseDiagnostics,
    LeasePool,
    LeasePoolCheckpoint,
    LeasePoolSnapshot,
)
from treemendous.domain import Span, validate_coordinate


class ProcessClock:
    """Monotonic millisecond clock for convenient process-local defaults."""

    def now(self) -> int:
        """Return monotonic process time in integer milliseconds."""
        return time.monotonic_ns() // 1_000_000


def require_string(value: str, name: str) -> str:
    """Validate one non-empty string."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def require_positive(value: int, name: str) -> int:
    """Validate one positive integer coordinate."""
    validate_coordinate(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def inclusive_span(start: int, end: int, name: str = "range") -> Span:
    """Convert a public inclusive integer range to a half-open Span."""
    validate_coordinate(start, f"{name} start")
    validate_coordinate(end, f"{name} end")
    if start > end:
        raise ValueError(f"{name} start must not exceed end")
    return Span(start, end + 1)


def spans_without(base: Span, excluded: Iterable[Span]) -> tuple[Span, ...]:
    """Return normalized portions of ``base`` outside the excluded spans."""
    clipped: list[Span] = []
    for span in excluded:
        if not isinstance(span, Span):
            raise TypeError("excluded ranges must contain Span values")
        start = max(base.start, span.start)
        end = min(base.end, span.end)
        if start < end:
            clipped.append(Span(start, end))
    clipped.sort(key=lambda item: (item.start, item.end))
    available: list[Span] = []
    cursor = base.start
    for span in clipped:
        if span.start > cursor:
            available.append(Span(cursor, span.start))
        cursor = max(cursor, span.end)
    if cursor < base.end:
        available.append(Span(cursor, base.end))
    if not available:
        raise ValueError("exclusions remove the complete leasing domain")
    return tuple(available)


@dataclass(frozen=True)
class NumericLease:
    """Domain-neutral wrapper retaining the stable pool scope of a lease."""

    scope: str
    lease: Lease

    def __post_init__(self) -> None:
        require_string(self.scope, "scope")
        if not isinstance(self.lease, Lease):
            raise TypeError("lease must be a Lease")

    @property
    def owner(self) -> str:
        """Return the owning principal."""
        return self.lease.owner

    @property
    def resource(self) -> Span:
        """Return the leased half-open numeric span."""
        return self.lease.resource

    @property
    def token(self) -> int:
        """Return the downstream fencing token."""
        return self.lease.token

    @property
    def pool_id(self) -> str:
        """Return the process-local pool lineage identifier."""
        return self.lease.pool_id

    @property
    def expires_at(self) -> int:
        """Return the exclusive logical expiry timestamp."""
        return self.lease.expires_at

    @property
    def revision(self) -> int:
        """Return the renewal revision."""
        return self.lease.revision


@dataclass(frozen=True)
class GroupSnapshot:
    """Immutable snapshots for every independently allocated scope."""

    pools: tuple[tuple[str, LeasePoolSnapshot], ...]


@dataclass(frozen=True)
class GroupCheckpoint:
    """Immutable checkpoints for every process-local pool lineage."""

    pools: tuple[tuple[str, LeasePoolCheckpoint], ...]


@dataclass(frozen=True)
class GroupDiagnostics:
    """Per-scope lease diagnostics."""

    pools: tuple[tuple[str, LeaseDiagnostics], ...]


class PoolGroup:
    """Small synchronized facade over scoped LeasePool instances.

    It is intentionally process-local. Checkpoint restoration creates fresh
    pool lineages, and callers must arrange single-writer takeover externally.
    """

    def __init__(self, domains: Mapping[str, tuple[Span, ...]], *, clock: Clock) -> None:
        if not domains:
            raise ValueError("at least one leasing scope is required")
        self.clock = clock
        self.pools = {
            require_string(scope, "scope"): LeasePool(spans, clock=clock)
            for scope, spans in domains.items()
        }
        self.fences = FenceValidator()
        self.lock = RLock()

    @classmethod
    def restore(cls, checkpoint: GroupCheckpoint, *, clock: Clock) -> PoolGroup:
        """Restore all pools into fresh process-local lineages."""
        if not isinstance(checkpoint, GroupCheckpoint):
            raise TypeError("checkpoint must be a GroupCheckpoint")
        if not checkpoint.pools:
            raise ValueError("checkpoint must contain at least one pool")
        group = cls.__new__(cls)
        group.clock = clock
        group.pools = {
            scope: LeasePool.from_checkpoint(pool_checkpoint, clock=clock)
            for scope, pool_checkpoint in checkpoint.pools
        }
        if len(group.pools) != len(checkpoint.pools):
            raise ValueError("checkpoint pool scopes must be unique")
        group.fences = FenceValidator()
        group.lock = RLock()
        return group

    def pool(self, scope: str) -> LeasePool:
        """Return a configured pool or reject an unknown domain scope."""
        try:
            return self.pools[scope]
        except KeyError:
            raise ValueError(f"unknown leasing scope: {scope!r}") from None

    def acquire(
        self,
        scope: str,
        owner: str,
        *,
        ttl: int,
        size: int = 1,
        exact_span: Span | None = None,
        request_id: str | None = None,
    ) -> NumericLease:
        """Acquire from one scope with LeasePool idempotency."""
        lease = self.pool(scope).acquire(
            owner,
            ttl=ttl,
            size=size,
            exact_span=exact_span,
            request_id=request_id,
        )
        return NumericLease(scope, lease)

    def renew(self, handle: NumericLease, *, ttl: int) -> NumericLease:
        """Renew a current handle and preserve its stable scope."""
        self._require_handle(handle)
        renewed = self.pool(handle.scope).renew(handle.lease, ttl=ttl)
        return NumericLease(handle.scope, renewed)

    def release(self, handle: NumericLease) -> NumericLease:
        """Release a current handle and preserve its stable scope."""
        self._require_handle(handle)
        released = self.pool(handle.scope).release(handle.lease)
        return NumericLease(handle.scope, released)

    @staticmethod
    def _require_handle(handle: NumericLease) -> None:
        if not isinstance(handle, NumericLease):
            raise TypeError("handle must be a NumericLease")

    def expire(self) -> tuple[NumericLease, ...]:
        """Materialize expirations across all scopes."""
        expired: list[NumericLease] = []
        for scope, pool in sorted(self.pools.items()):
            expired.extend(NumericLease(scope, lease) for lease in pool.expire())
        return tuple(expired)

    def validate_fence(self, key: Hashable, handle: NumericLease) -> bool:
        """Validate an issued handle against one stable downstream key."""
        self._require_handle(handle)
        pool = self.pool(handle.scope)
        if handle.pool_id != pool.pool_id:
            raise ForeignLeaseError("lease belongs to another pool lineage")
        issued = next(
            (
                lease
                for lease in pool.snapshot().leases
                if lease.token == handle.token
            ),
            None,
        )
        if issued is None:
            raise InvalidLeaseError("lease token was not issued by this pool")
        if (
            issued.owner != handle.owner
            or issued.resource != handle.resource
            or issued.acquired_at != handle.lease.acquired_at
            or issued.request_id != handle.lease.request_id
        ):
            raise InvalidLeaseError("lease evidence differs from issued history")
        identity = (handle.pool_id, handle.token)
        return self.fences.validate_fence(
            key,
            handle.token,
            lease_identity=identity,
        )

    def snapshot(self) -> GroupSnapshot:
        """Capture every pool in deterministic scope order."""
        return GroupSnapshot(
            tuple((scope, pool.snapshot()) for scope, pool in sorted(self.pools.items()))
        )

    def checkpoint(self) -> GroupCheckpoint:
        """Capture restorable pool state, excluding process-local fence state."""
        return GroupCheckpoint(
            tuple(
                (scope, pool.checkpoint())
                for scope, pool in sorted(self.pools.items())
            )
        )

    def diagnostics(self) -> GroupDiagnostics:
        """Return deterministic per-scope operational diagnostics."""
        return GroupDiagnostics(
            tuple(
                (scope, pool.diagnostics())
                for scope, pool in sorted(self.pools.items())
            )
        )
