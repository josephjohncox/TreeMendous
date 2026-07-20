"""Monotonic database ID batches with explicit commit/reuse semantics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from treemendous.applications._shared.clock import Clock
from treemendous.applications._shared.leasing import (
    LeaseRequestConflictError,
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

DatabaseIdLease = NumericLease


class DatabaseIdUnavailableError(LeaseUnavailableError):
    """Raised when no monotonic or reusable ID batch satisfies a request."""


class CommittedIdError(RuntimeError):
    """Raised when a committed permanent ID batch is used as a lease."""


@dataclass(frozen=True)
class CommittedIdBatch:
    """Permanent IDs separated from the renewable/reusable lease lifecycle."""

    namespace: str
    source: DatabaseIdLease
    committed_at: int

    @property
    def resource(self) -> Span:
        """Return the permanently committed half-open ID span."""
        return self.source.resource

    @property
    def token(self) -> int:
        """Return the source lease's stable fencing token."""
        return self.source.token


@dataclass(frozen=True)
class _DatabaseRequest:
    request_id: str
    owner: str
    ttl: int
    count: int
    reusable: bool
    resource: Span


@dataclass(frozen=True)
class DatabaseIdCheckpoint:
    """Complete local allocator state, including permanent commit evidence."""

    namespace: str
    minimum_id: int
    maximum_id: int
    next_monotonic_id: int
    reusable_spans: tuple[Span, ...]
    committed: tuple[CommittedIdBatch, ...]
    requests: tuple[_DatabaseRequest, ...]
    group: GroupCheckpoint


@dataclass(frozen=True)
class DatabaseIdSnapshot:
    """Observable distinction between pool leases and committed ID batches."""

    next_monotonic_id: int
    reusable_spans: tuple[Span, ...]
    committed: tuple[CommittedIdBatch, ...]
    pools: GroupSnapshot


def _remove_span(spans: list[Span], selected: Span) -> list[Span]:
    result: list[Span] = []
    for span in spans:
        if not span.overlaps(selected):
            result.append(span)
            continue
        if span.start < selected.start:
            result.append(Span(span.start, selected.start))
        if selected.end < span.end:
            result.append(Span(selected.end, span.end))
    return sorted(result, key=lambda item: (item.start, item.end))


def _merge_spans(spans: list[Span]) -> list[Span]:
    merged: list[Span] = []
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if not merged or merged[-1].end < span.start:
            merged.append(span)
        else:
            previous = merged[-1]
            merged[-1] = Span(previous.start, max(previous.end, span.end))
    return merged


class DatabaseIdPool:
    """Issue monotonic temporary batches and explicitly commit or recycle them.

    Normal acquisitions always advance ``next_monotonic_id``. A released or
    expired uncommitted batch enters an explicit reusable queue and is used only
    with ``reusable=True``. ``commit`` permanently records IDs and removes their
    TTL lifecycle. This is a process-local allocation engine, not a database
    transaction or a durable sequence service.
    """

    def __init__(
        self,
        namespace: str = "default",
        *,
        minimum_id: int = 1,
        maximum_id: int = 2**63 - 1,
        clock: Clock | None = None,
    ) -> None:
        self.namespace = require_string(namespace, "namespace")
        validate_coordinate(minimum_id, "minimum_id")
        validate_coordinate(maximum_id, "maximum_id")
        if minimum_id > maximum_id:
            raise ValueError("minimum_id must not exceed maximum_id")
        self.minimum_id = minimum_id
        self.maximum_id = maximum_id
        self._next_id = minimum_id
        self._reusable: list[Span] = []
        self._committed: dict[int, CommittedIdBatch] = {}
        self._requests: dict[str, _DatabaseRequest] = {}
        self._group = PoolGroup(
            {self.namespace: (inclusive_span(minimum_id, maximum_id, "ID domain"),)},
            clock=clock if clock is not None else ProcessClock(),
        )

    @classmethod
    def from_checkpoint(
        cls, checkpoint: DatabaseIdCheckpoint, *, clock: Clock
    ) -> DatabaseIdPool:
        """Restore state into a fresh local lineage after external takeover."""
        if not isinstance(checkpoint, DatabaseIdCheckpoint):
            raise TypeError("checkpoint must be a DatabaseIdCheckpoint")
        engine = cls.__new__(cls)
        engine.namespace = require_string(checkpoint.namespace, "namespace")
        engine.minimum_id = checkpoint.minimum_id
        engine.maximum_id = checkpoint.maximum_id
        engine._next_id = checkpoint.next_monotonic_id
        engine._reusable = list(checkpoint.reusable_spans)
        engine._requests = {record.request_id: record for record in checkpoint.requests}
        if len(engine._requests) != len(checkpoint.requests):
            raise ValueError("checkpoint request IDs must be unique")
        engine._group = PoolGroup.restore(checkpoint.group, clock=clock)
        if set(engine._group.pools) != {engine.namespace}:
            raise ValueError("database checkpoint namespace does not match its pool")
        leases_by_token = {
            lease.token: lease
            for lease in engine._group.pool(engine.namespace).snapshot().leases
        }
        engine._committed = {}
        for committed in checkpoint.committed:
            restored_lease = leases_by_token.get(committed.token)
            if restored_lease is None:
                raise ValueError("committed batch source is absent from checkpoint")
            restored = replace(
                committed,
                source=NumericLease(engine.namespace, restored_lease),
            )
            engine._committed[restored.token] = restored
        if len(engine._committed) != len(checkpoint.committed):
            raise ValueError("checkpoint committed tokens must be unique")
        return engine

    def _collect_expired(self) -> tuple[DatabaseIdLease, ...]:
        with self._group.lock:
            expired = self._group.expire()
            for handle in expired:
                if handle.token not in self._committed:
                    self._reusable = _merge_spans(
                        [*self._reusable, handle.resource]
                    )
            return expired

    def acquire(
        self,
        owner: str,
        *,
        ttl: int,
        count: int = 1,
        reusable: bool = False,
        request_id: str | None = None,
    ) -> DatabaseIdLease:
        """Acquire a monotonic batch, or explicitly consume recycled capacity."""
        count = require_positive(count, "count")
        self._collect_expired()
        with self._group.lock:
            existing = None if request_id is None else self._requests.get(request_id)
            if existing is not None:
                fingerprint = (owner, ttl, count, reusable)
                recorded = (
                    existing.owner,
                    existing.ttl,
                    existing.count,
                    existing.reusable,
                )
                if fingerprint != recorded:
                    raise LeaseRequestConflictError(
                        f"request_id {request_id!r} was already used differently"
                    )
                return self._group.acquire(
                    self.namespace,
                    owner,
                    ttl=ttl,
                    size=count,
                    exact_span=existing.resource,
                    request_id=request_id,
                )

            selected = None
            if reusable:
                selected = next(
                    (
                        Span(span.start, span.start + count)
                        for span in self._reusable
                        if span.length >= count
                    ),
                    None,
                )
                if selected is None:
                    raise DatabaseIdUnavailableError(
                        "no reusable uncommitted batch is available"
                    )
            else:
                end = self._next_id + count
                if end > self.maximum_id + 1:
                    raise DatabaseIdUnavailableError("monotonic ID space is exhausted")
                selected = Span(self._next_id, end)
            try:
                handle = self._group.acquire(
                    self.namespace,
                    owner,
                    ttl=ttl,
                    size=count,
                    exact_span=selected,
                    request_id=request_id,
                )
            except LeaseUnavailableError as exc:
                raise DatabaseIdUnavailableError(str(exc)) from None
            if reusable:
                self._reusable = _remove_span(self._reusable, selected)
            else:
                self._next_id = selected.end
            if request_id is not None:
                self._requests[request_id] = _DatabaseRequest(
                    request_id,
                    owner,
                    ttl,
                    count,
                    reusable,
                    selected,
                )
            return handle

    def renew(self, handle: DatabaseIdLease, *, ttl: int) -> DatabaseIdLease:
        """Renew an uncommitted current ID lease."""
        with self._group.lock:
            if handle.token in self._committed:
                raise CommittedIdError(
                    "committed IDs are permanent, not renewable leases"
                )
            return self._group.renew(handle, ttl=ttl)

    def release(self, handle: DatabaseIdLease) -> DatabaseIdLease:
        """Release an uncommitted batch into the explicit reusable queue."""
        with self._group.lock:
            if handle.token in self._committed:
                raise CommittedIdError("committed IDs cannot be released")
            released = self._group.release(handle)
            self._reusable = _merge_spans([*self._reusable, released.resource])
            return released

    def commit(self, handle: DatabaseIdLease) -> CommittedIdBatch:
        """Make a current batch permanent and end its renewable lease state."""
        with self._group.lock:
            existing = self._committed.get(handle.token)
            if existing is not None:
                if existing.source != handle:
                    raise CommittedIdError("commit retry used altered lease evidence")
                return existing
            committed_at = self._group.clock.now()
            self._group.release(handle)
            committed = CommittedIdBatch(self.namespace, handle, committed_at)
            self._committed[handle.token] = committed
            return committed

    def expire(self) -> tuple[DatabaseIdLease, ...]:
        """Expire temporary batches and expose them for explicit reuse."""
        return self._collect_expired()

    def validate_fence(
        self,
        handle: DatabaseIdLease | CommittedIdBatch,
        identifier: int,
    ) -> bool:
        """Fence one ID under a stable namespace/identifier key."""
        validate_coordinate(identifier, "identifier")
        source = handle.source if isinstance(handle, CommittedIdBatch) else handle
        if identifier < source.resource.start or identifier >= source.resource.end:
            raise ValueError("identifier is outside the batch")
        key = ("database-id-pools", self.namespace, identifier)
        return self._group.validate_fence(key, source)

    def snapshot(self) -> DatabaseIdSnapshot:
        """Return leases, reusable spans, monotonic cursor, and permanent IDs."""
        with self._group.lock:
            self._collect_expired()
            return DatabaseIdSnapshot(
                self._next_id,
                tuple(self._reusable),
                tuple(self._committed[token] for token in sorted(self._committed)),
                self._group.snapshot(),
            )

    def checkpoint(self) -> DatabaseIdCheckpoint:
        """Return complete allocator state, excluding downstream fence marks."""
        with self._group.lock:
            snapshot = self.snapshot()
            return DatabaseIdCheckpoint(
                self.namespace,
                self.minimum_id,
                self.maximum_id,
                snapshot.next_monotonic_id,
                snapshot.reusable_spans,
                snapshot.committed,
                tuple(self._requests[key] for key in sorted(self._requests)),
                self._group.checkpoint(),
            )

    def diagnostics(self) -> GroupDiagnostics:
        """Return lifecycle and capacity counters for the underlying lease pool."""
        with self._group.lock:
            self._collect_expired()
            return self._group.diagnostics()


def create_engine(**kwargs: Any) -> DatabaseIdPool:
    """Create the manifest factory for database ID pools."""
    return DatabaseIdPool(**kwargs)
