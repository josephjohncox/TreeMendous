"""Private deterministic numeric leases with local fencing tokens.

The kernel serializes in-process state transitions and can be checkpointed, but
it is not a consensus system.  A fencing token only becomes effective when the
downstream resource durably rejects older tokens; expiry cannot stop an old
holder from continuing to run.
"""

from __future__ import annotations

import secrets
from collections.abc import Hashable
from dataclasses import dataclass, replace
from enum import Enum
from threading import RLock

from treemendous.applications._shared.clock import Clock
from treemendous.backends.adapters import BackendAdapter
from treemendous.basic.boundary import IntervalManager
from treemendous.domain import (
    DomainInput,
    ManagedDomain,
    Span,
    validate_coordinate,
)
from treemendous.rangeset import RangeSet


class LeaseError(RuntimeError):
    """Base error raised by the private lease kernel."""


class LeaseUnavailableError(LeaseError):
    """Raised when the requested resource is not currently available."""


class LeaseRequestConflictError(LeaseError):
    """Raised when a request ID is reused with different arguments."""


class InvalidLeaseError(LeaseError):
    """Raised when a handle does not identify a lease in this pool."""


class ForeignLeaseError(InvalidLeaseError):
    """Raised when a handle belongs to another pool or owner."""


class StaleLeaseError(InvalidLeaseError):
    """Raised when an older revision or released handle is used."""


class ExpiredLeaseError(StaleLeaseError):
    """Raised when an expired lease is renewed or released."""


class LeaseState(Enum):
    """Explicit lifecycle state retained in snapshots and checkpoints."""

    ACTIVE = "active"
    EXPIRED = "expired"
    RELEASED = "released"


def _validate_nonempty_string(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _validate_positive(value: int, name: str) -> int:
    validate_coordinate(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _coerce_span(value: Span | tuple[int, int], name: str) -> Span:
    if isinstance(value, Span):
        return value
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a Span or (start, end) tuple")
    try:
        start, end = value
        return Span(start, end)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"invalid {name}: {exc}") from exc


@dataclass(frozen=True)
class Lease:
    """Immutable, owner-scoped evidence for one lease revision."""

    pool_id: str
    owner: str
    resource: Span
    fencing_token: int
    acquired_at: int
    expires_at: int
    state: LeaseState = LeaseState.ACTIVE
    revision: int = 1
    request_id: str | None = None

    def __post_init__(self) -> None:
        _validate_nonempty_string(self.pool_id, "pool_id")
        _validate_nonempty_string(self.owner, "owner")
        _validate_positive(self.fencing_token, "fencing_token")
        validate_coordinate(self.acquired_at, "acquired_at")
        validate_coordinate(self.expires_at, "expires_at")
        if self.expires_at <= self.acquired_at:
            raise ValueError("expires_at must be greater than acquired_at")
        if not isinstance(self.state, LeaseState):
            raise TypeError("state must be a LeaseState")
        _validate_positive(self.revision, "revision")
        if self.request_id is not None:
            _validate_nonempty_string(self.request_id, "request_id")

    @property
    def span(self) -> Span:
        """Alias the leased numeric resource for range-oriented callers."""
        return self.resource

    @property
    def token(self) -> int:
        """Return the globally ordered fencing token within this pool lineage."""
        return self.fencing_token

    @property
    def active(self) -> bool:
        """Return whether this recorded revision has active state."""
        return self.state is LeaseState.ACTIVE


@dataclass(frozen=True)
class _RequestRecord:
    request_id: str
    owner: str
    ttl: int
    size: int
    alignment: int
    exact_span: Span | None
    lease_token: int

    def __post_init__(self) -> None:
        _validate_nonempty_string(self.request_id, "request_id")
        _validate_nonempty_string(self.owner, "owner")
        _validate_positive(self.ttl, "ttl")
        _validate_positive(self.size, "size")
        _validate_positive(self.alignment, "alignment")
        _validate_positive(self.lease_token, "lease_token")


@dataclass(frozen=True)
class LeaseDiagnostics:
    """Small immutable operational summary of a lease pool."""

    observed_at: int
    total_capacity: int
    available_capacity: int
    largest_available_span: int
    active_leases: int
    expired_leases: int
    released_leases: int
    issued_tokens: int
    next_fencing_token: int


@dataclass(frozen=True)
class LeasePoolSnapshot:
    """Immutable observable pool state, including terminal lease records."""

    observed_at: int
    pool_id: str
    allowed_spans: tuple[Span, ...]
    available_spans: tuple[Span, ...]
    leases: tuple[Lease, ...]
    diagnostics: LeaseDiagnostics


@dataclass(frozen=True)
class LeasePoolCheckpoint:
    """Complete durable state needed to restore a pool without token reuse."""

    pool_id: str
    allowed_spans: tuple[Span, ...]
    leases: tuple[Lease, ...]
    requests: tuple[_RequestRecord, ...]
    next_fencing_token: int
    last_observed_at: int

    def __post_init__(self) -> None:
        _validate_nonempty_string(self.pool_id, "pool_id")
        if not self.allowed_spans:
            raise ValueError("allowed_spans must contain at least one span")
        _validate_positive(self.next_fencing_token, "next_fencing_token")
        validate_coordinate(self.last_observed_at, "last_observed_at")


class LeasePool:
    """Thread-safe deterministic allocator for fenced integer-span leases.

    Tokens increase across every resource in a pool and are preserved across
    :meth:`checkpoint` / :meth:`from_checkpoint`.  Request IDs are retained for
    the checkpoint's lifetime, so retrying an expired or released request
    returns that terminal lease rather than allocating a new token.
    """

    def __init__(self, allowed_spans: DomainInput, *, clock: Clock) -> None:
        if not hasattr(clock, "now") or not callable(clock.now):
            raise TypeError("clock must provide a callable now()")
        domain = (
            allowed_spans
            if isinstance(allowed_spans, ManagedDomain)
            else ManagedDomain(allowed_spans)
        )
        observed_at = self._read_clock(clock)
        self._clock = clock
        self._domain = domain
        self._pool_id = secrets.token_hex(16)
        self._leases: dict[int, Lease] = {}
        self._requests: dict[str, _RequestRecord] = {}
        self._next_fencing_token = 1
        self._last_observed_at = observed_at
        self._free = self._build_free(self._leases)
        self._lock = RLock()

    @staticmethod
    def _read_clock(clock: Clock) -> int:
        return validate_coordinate(clock.now(), "clock.now()")

    @property
    def allowed_spans(self) -> tuple[Span, ...]:
        """Return the normalized immutable allocation domain."""
        return self._domain.spans

    @property
    def pool_id(self) -> str:
        """Return the identity embedded in handles and checkpoints."""
        return self._pool_id

    def _observe_time(self) -> int:
        now = self._read_clock(self._clock)
        if now < self._last_observed_at:
            raise RuntimeError("lease clock cannot move backwards")
        return now

    def _new_range_set(self) -> RangeSet:
        return RangeSet(
            BackendAdapter(IntervalManager()),
            domain=self._domain,
            initially_available=True,
        )

    def _build_free(self, leases: dict[int, Lease]) -> RangeSet:
        free = self._new_range_set()
        active = sorted(
            (lease for lease in leases.values() if lease.state is LeaseState.ACTIVE),
            key=lambda lease: (lease.resource.start, lease.resource.end, lease.token),
        )
        for lease in active:
            mutation = free.discard(lease.resource, require_covered=True)
            if not mutation.fully_covered:
                raise ValueError(
                    "active leases must be disjoint and inside allowed_spans"
                )
        return free

    def _stage_expirations(
        self, now: int
    ) -> tuple[dict[int, Lease], RangeSet, tuple[Lease, ...]]:
        expired = tuple(
            lease
            for lease in self._leases.values()
            if lease.state is LeaseState.ACTIVE and lease.expires_at <= now
        )
        if not expired:
            return self._leases, self._free, ()
        staged = dict(self._leases)
        transitioned: list[Lease] = []
        for lease in sorted(expired, key=lambda item: item.token):
            terminal = replace(lease, state=LeaseState.EXPIRED)
            staged[lease.token] = terminal
            transitioned.append(terminal)
        return staged, self._build_free(staged), tuple(transitioned)

    def _commit(
        self, leases: dict[int, Lease], free: RangeSet, observed_at: int
    ) -> None:
        self._leases = leases
        self._free = free
        self._last_observed_at = observed_at

    @staticmethod
    def _aligned_at_or_after(value: int, alignment: int) -> int:
        return -(-value // alignment) * alignment

    def _select_span(
        self, free: RangeSet, *, size: int, alignment: int, exact_span: Span | None
    ) -> Span:
        if exact_span is not None:
            if not self._domain.contains(exact_span):
                raise LeaseUnavailableError("exact_span is outside allowed_spans")
            if exact_span.start % alignment:
                raise ValueError("exact_span start must satisfy alignment")
            mutation = free.discard(exact_span, require_covered=True)
            if not mutation.fully_covered:
                raise LeaseUnavailableError("exact_span is not available")
            return exact_span

        for interval in free.intervals():
            start = self._aligned_at_or_after(interval.start, alignment)
            candidate = Span(start, start + size)
            if candidate.end <= interval.end:
                mutation = free.discard(candidate, require_covered=True)
                if not mutation.fully_covered:
                    raise RuntimeError("deterministic allocation lost free coverage")
                return candidate
        raise LeaseUnavailableError(
            "no aligned span of the requested size is available"
        )

    def acquire(
        self,
        owner: str,
        *,
        ttl: int,
        size: int | None = None,
        alignment: int = 1,
        exact_span: Span | tuple[int, int] | None = None,
        request_id: str | None = None,
    ) -> Lease:
        """Acquire the earliest aligned span, or one exact span.

        Identical retries with ``request_id`` return the originally issued
        lease, including its current terminal state.  Reuse with any different
        normalized argument raises :class:`LeaseRequestConflictError`.
        """
        owner = _validate_nonempty_string(owner, "owner")
        ttl = _validate_positive(ttl, "ttl")
        alignment = _validate_positive(alignment, "alignment")
        requested_exact = (
            None if exact_span is None else _coerce_span(exact_span, "exact_span")
        )
        if size is None:
            effective_size = (
                requested_exact.length if requested_exact is not None else 1
            )
        else:
            effective_size = _validate_positive(size, "size")
        if requested_exact is not None and requested_exact.length != effective_size:
            raise ValueError("size must equal exact_span length")
        if request_id is not None:
            request_id = _validate_nonempty_string(request_id, "request_id")

        with self._lock:
            now = self._observe_time()
            staged, staged_free, _ = self._stage_expirations(now)
            fingerprint = (
                owner,
                ttl,
                effective_size,
                alignment,
                requested_exact,
            )
            existing = None if request_id is None else self._requests.get(request_id)
            if existing is not None:
                existing_fingerprint = (
                    existing.owner,
                    existing.ttl,
                    existing.size,
                    existing.alignment,
                    existing.exact_span,
                )
                if fingerprint != existing_fingerprint:
                    raise LeaseRequestConflictError(
                        f"request_id {request_id!r} was already used differently"
                    )
                lease = staged[existing.lease_token]
                self._commit(staged, staged_free, now)
                return lease

            # Allocate against a staged RangeSet even when no expiry occurred;
            # the live free set is untouched until the complete lease record is
            # ready to commit.
            allocation_free = self._build_free(staged)
            resource = self._select_span(
                allocation_free,
                size=effective_size,
                alignment=alignment,
                exact_span=requested_exact,
            )
            token = self._next_fencing_token
            lease = Lease(
                self._pool_id,
                owner,
                resource,
                token,
                now,
                now + ttl,
                request_id=request_id,
            )
            committed = dict(staged)
            committed[token] = lease
            committed_requests = self._requests
            if request_id is not None:
                committed_requests = dict(self._requests)
                committed_requests[request_id] = _RequestRecord(
                    request_id,
                    owner,
                    ttl,
                    effective_size,
                    alignment,
                    requested_exact,
                    token,
                )
            self._commit(committed, allocation_free, now)
            self._requests = committed_requests
            self._next_fencing_token = token + 1
            return lease

    def _current_for(
        self,
        handle: Lease,
        leases: dict[int, Lease],
        *,
        owner: str | None,
    ) -> Lease:
        if not isinstance(handle, Lease):
            raise TypeError("handle must be a Lease")
        if handle.pool_id != self._pool_id:
            raise ForeignLeaseError("lease belongs to another pool")
        asserted_owner = (
            handle.owner if owner is None else _validate_nonempty_string(owner, "owner")
        )
        if asserted_owner != handle.owner:
            raise ForeignLeaseError("lease belongs to another owner")
        current = leases.get(handle.token)
        if current is None:
            raise InvalidLeaseError("lease token was not issued by this pool")
        if current.owner != asserted_owner:
            raise ForeignLeaseError("lease belongs to another owner")
        if current.state is LeaseState.EXPIRED:
            raise ExpiredLeaseError("lease has expired")
        if current.state is LeaseState.RELEASED:
            raise StaleLeaseError("lease has already been released")
        if current != handle:
            raise StaleLeaseError("lease handle is an old or altered revision")
        return current

    def renew(self, handle: Lease, *, ttl: int, owner: str | None = None) -> Lease:
        """Renew an active current handle from now and return its next revision."""
        ttl = _validate_positive(ttl, "ttl")
        with self._lock:
            now = self._observe_time()
            staged, staged_free, _ = self._stage_expirations(now)
            current = self._current_for(handle, staged, owner=owner)
            renewed = replace(
                current,
                expires_at=now + ttl,
                revision=current.revision + 1,
            )
            committed = dict(staged)
            committed[current.token] = renewed
            self._commit(committed, staged_free, now)
            return renewed

    def release(self, handle: Lease, *, owner: str | None = None) -> Lease:
        """Release an active current handle and return its terminal record."""
        with self._lock:
            now = self._observe_time()
            staged, _, _ = self._stage_expirations(now)
            current = self._current_for(handle, staged, owner=owner)
            released = replace(current, state=LeaseState.RELEASED)
            committed = dict(staged)
            committed[current.token] = released
            free = self._build_free(committed)
            self._commit(committed, free, now)
            return released

    def expire(self) -> tuple[Lease, ...]:
        """Materialize clock-expired leases and return newly expired records."""
        with self._lock:
            now = self._observe_time()
            staged, free, expired = self._stage_expirations(now)
            self._commit(staged, free, now)
            return expired

    def _diagnostics_at(
        self, now: int, leases: dict[int, Lease], free: RangeSet
    ) -> LeaseDiagnostics:
        intervals = free.intervals()
        states = tuple(lease.state for lease in leases.values())
        return LeaseDiagnostics(
            observed_at=now,
            total_capacity=self._domain.measure,
            available_capacity=sum(
                interval.end - interval.start for interval in intervals
            ),
            largest_available_span=max(
                (interval.end - interval.start for interval in intervals), default=0
            ),
            active_leases=states.count(LeaseState.ACTIVE),
            expired_leases=states.count(LeaseState.EXPIRED),
            released_leases=states.count(LeaseState.RELEASED),
            issued_tokens=self._next_fencing_token - 1,
            next_fencing_token=self._next_fencing_token,
        )

    def diagnostics(self) -> LeaseDiagnostics:
        """Return capacity, lifecycle, and token counters at the current time."""
        with self._lock:
            now = self._observe_time()
            staged, free, _ = self._stage_expirations(now)
            self._commit(staged, free, now)
            return self._diagnostics_at(now, staged, free)

    def snapshot(self) -> LeasePoolSnapshot:
        """Return an immutable, time-consistent observable snapshot."""
        with self._lock:
            now = self._observe_time()
            staged, free, _ = self._stage_expirations(now)
            leases = tuple(sorted(staged.values(), key=lambda lease: lease.token))
            available = tuple(interval.span for interval in free.intervals())
            diagnostics = self._diagnostics_at(now, staged, free)
            result = LeasePoolSnapshot(
                now,
                self._pool_id,
                self._domain.spans,
                available,
                leases,
                diagnostics,
            )
            self._commit(staged, free, now)
            return result

    def checkpoint(self) -> LeasePoolCheckpoint:
        """Return complete immutable state suitable for exact restoration."""
        with self._lock:
            now = self._observe_time()
            staged, free, _ = self._stage_expirations(now)
            result = LeasePoolCheckpoint(
                pool_id=self._pool_id,
                allowed_spans=self._domain.spans,
                leases=tuple(sorted(staged.values(), key=lambda lease: lease.token)),
                requests=tuple(
                    sorted(self._requests.values(), key=lambda item: item.request_id)
                ),
                next_fencing_token=self._next_fencing_token,
                last_observed_at=now,
            )
            self._commit(staged, free, now)
            return result

    @classmethod
    def from_checkpoint(
        cls, checkpoint: LeasePoolCheckpoint, *, clock: Clock
    ) -> LeasePool:
        """Restore validated state while preserving the next fencing token."""
        if not isinstance(checkpoint, LeasePoolCheckpoint):
            raise TypeError("checkpoint must be a LeasePoolCheckpoint")
        if not hasattr(clock, "now") or not callable(clock.now):
            raise TypeError("clock must provide a callable now()")
        domain = ManagedDomain(checkpoint.allowed_spans)
        if domain.spans != checkpoint.allowed_spans:
            raise ValueError("checkpoint allowed_spans must already be normalized")
        now = cls._read_clock(clock)
        if now < checkpoint.last_observed_at:
            raise ValueError("restore clock cannot precede checkpoint time")

        leases: dict[int, Lease] = {}
        for lease in checkpoint.leases:
            if not isinstance(lease, Lease):
                raise TypeError("checkpoint leases must contain Lease values")
            if lease.pool_id != checkpoint.pool_id:
                raise ValueError("checkpoint lease belongs to another pool")
            if not domain.contains(lease.resource):
                raise ValueError("checkpoint lease is outside allowed_spans")
            if lease.token in leases:
                raise ValueError("checkpoint lease tokens must be unique")
            leases[lease.token] = lease
        if leases and checkpoint.next_fencing_token <= max(leases):
            raise ValueError("next_fencing_token must exceed every issued token")

        requests: dict[str, _RequestRecord] = {}
        for record in checkpoint.requests:
            if not isinstance(record, _RequestRecord):
                raise TypeError("checkpoint requests contain an invalid record")
            if record.request_id in requests:
                raise ValueError("checkpoint request IDs must be unique")
            request_lease = leases.get(record.lease_token)
            if request_lease is None or request_lease.request_id != record.request_id:
                raise ValueError("checkpoint request does not identify its lease")
            if (
                request_lease.owner != record.owner
                or request_lease.resource.length != record.size
            ):
                raise ValueError("checkpoint request conflicts with its lease")
            if (
                record.exact_span is not None
                and record.exact_span != request_lease.resource
            ):
                raise ValueError("checkpoint exact request conflicts with its lease")
            requests[record.request_id] = record
        for lease in leases.values():
            if lease.request_id is not None and lease.request_id not in requests:
                raise ValueError("checkpoint lease is missing its request record")

        pool = cls.__new__(cls)
        pool._clock = clock
        pool._domain = domain
        pool._pool_id = checkpoint.pool_id
        pool._requests = requests
        pool._next_fencing_token = checkpoint.next_fencing_token
        pool._last_observed_at = now
        pool._lock = RLock()
        transitioned = {
            token: (
                replace(lease, state=LeaseState.EXPIRED)
                if lease.state is LeaseState.ACTIVE and lease.expires_at <= now
                else lease
            )
            for token, lease in leases.items()
        }
        pool._leases = transitioned
        pool._free = pool._build_free(transitioned)
        return pool


class FenceValidator:
    """Thread-safe downstream high-water-mark check for fencing tokens.

    ``validate_fence`` accepts equal tokens for idempotent retries, advances the
    resource's high-water mark for newer tokens, and rejects older tokens.  It
    is only local process state: it provides neither consensus nor revocation,
    and it does not make an expired holder stop.  A real resource must persist
    and enforce the high-water mark at the write it protects.
    """

    def __init__(self) -> None:
        self._highest: dict[Hashable, int] = {}
        self._lock = RLock()

    def validate_fence(self, resource: Hashable, token: int) -> bool:
        """Return whether ``token`` is at least the resource's accepted maximum."""
        token = _validate_positive(token, "token")
        try:
            hash(resource)
        except TypeError as exc:
            raise TypeError("resource must be hashable") from exc
        with self._lock:
            highest = self._highest.get(resource)
            if highest is not None and token < highest:
                return False
            self._highest[resource] = token
            return True

    def highest_token(self, resource: Hashable) -> int | None:
        """Return the accepted high-water mark for one downstream resource."""
        try:
            hash(resource)
        except TypeError as exc:
            raise TypeError("resource must be hashable") from exc
        with self._lock:
            return self._highest.get(resource)
