"""Private deterministic work claiming with local fencing evidence.

This state machine coordinates finite integer work ranges in one process.  Its
checkpoint is serializable application state, not durable storage, consensus,
worker liveness, or proof that an expired/abandoned worker stopped executing.
Downstream result commits must durably enforce fencing tokens when stale workers
can still write.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, replace
from enum import Enum
from threading import RLock
from typing import Any

from treemendous.applications._shared.clock import Clock
from treemendous.applications._shared.events import (
    Event,
    EventLog,
    EventLogCheckpoint,
    EventLogSnapshot,
    freeze_metadata,
)
from treemendous.backends.adapters import BackendAdapter
from treemendous.basic.boundary import IntervalManager
from treemendous.domain import (
    DomainInput,
    ManagedDomain,
    Span,
    validate_coordinate,
    validate_length,
)
from treemendous.rangeset import RangeSet


class ClaimError(RuntimeError):
    """Base error for work-claim transitions."""


class ClaimUnavailableError(ClaimError):
    """Raised when no contiguous work range can satisfy a request."""


class ClaimRequestConflictError(ClaimError):
    """Raised when a request ID is reused with different arguments."""


class InvalidClaimError(ClaimError):
    """Raised when a handle does not identify a claim in this ledger."""


class ForeignClaimError(InvalidClaimError):
    """Raised for a different ledger or owner."""


class StaleClaimError(InvalidClaimError):
    """Raised for an older revision/fencing token."""


class TerminalClaimError(StaleClaimError):
    """Raised when a terminal claim is used as active."""


class ExpiredClaimError(TerminalClaimError):
    """Raised when a claim expired before a requested transition."""


class ClaimInvariantError(ClaimError):
    """Raised when restored or live state violates ledger invariants."""


class ClaimState(Enum):
    """Explicit work lifecycle retained for audit/checkpointing."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    EXPIRED = "expired"


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _positive(value: int, name: str) -> int:
    validate_coordinate(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _positive_or_none(value: int | None, name: str) -> int | None:
    return None if value is None else _positive(value, name)


@dataclass(frozen=True)
class WorkClaim:
    """Immutable owner-scoped evidence for one claim revision."""

    ledger_id: str
    claim_id: int
    owner: str
    span: Span
    fencing_token: int
    acquired_at: int
    expires_at: int | None
    state: ClaimState = ClaimState.ACTIVE
    revision: int = 1
    request_id: str | None = None
    result: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.ledger_id, "ledger_id")
        _nonempty(self.owner, "owner")
        validate_coordinate(self.claim_id, "claim_id")
        validate_coordinate(self.fencing_token, "fencing_token")
        validate_coordinate(self.acquired_at, "acquired_at")
        validate_coordinate(self.revision, "revision")
        if min(self.claim_id, self.fencing_token, self.revision) <= 0:
            raise ValueError("claim_id, fencing_token, and revision must be positive")
        if self.expires_at is not None:
            validate_coordinate(self.expires_at, "expires_at")
            if self.expires_at <= self.acquired_at:
                raise ValueError("expires_at must be greater than acquired_at")
        if not isinstance(self.state, ClaimState):
            raise TypeError("state must be a ClaimState")
        if self.request_id is not None:
            _nonempty(self.request_id, "request_id")
        if not isinstance(self.result, tuple):
            raise TypeError("result must be a tuple of key/value pairs")
        result_mapping: dict[str, Any] = {}
        for item in self.result:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("result must contain key/value pairs")
            key, value = item
            if not isinstance(key, str):
                raise TypeError("result mapping keys must be strings")
            if key in result_mapping:
                raise ValueError("result keys must be unique")
            result_mapping[key] = value
        object.__setattr__(self, "result", freeze_metadata(result_mapping))
        if self.state is not ClaimState.COMPLETED and self.result:
            raise ValueError("only completed claims may contain result metadata")

    @property
    def active(self) -> bool:
        return self.state is ClaimState.ACTIVE


@dataclass(frozen=True)
class _ClaimRequest:
    request_id: str
    owner: str
    length: int
    not_before: int
    not_after: int | None
    ttl: int | None
    claim_id: int

    def __post_init__(self) -> None:
        _nonempty(self.request_id, "request_id")
        _nonempty(self.owner, "owner")
        validate_length(self.length)
        validate_coordinate(self.not_before, "not_before")
        if self.not_after is not None:
            validate_coordinate(self.not_after, "not_after")
            if self.not_after <= self.not_before:
                raise ValueError("not_after must be greater than not_before")
        _positive_or_none(self.ttl, "ttl")
        validate_coordinate(self.claim_id, "claim_id")
        if self.claim_id <= 0:
            raise ValueError("claim_id must be positive")


@dataclass(frozen=True)
class ClaimDiagnostics:
    """Immutable capacity/lifecycle/fencing summary."""

    observed_at: int
    total_work: int
    available_work: int
    active_claims: int
    completed_claims: int
    abandoned_claims: int
    expired_claims: int
    issued_claim_ids: int
    issued_fencing_tokens: int
    next_claim_id: int
    next_fencing_token: int


@dataclass(frozen=True)
class ClaimLedgerSnapshot:
    """Deterministic point-in-time ledger observation."""

    observed_at: int
    ledger_id: str
    domain: ManagedDomain
    available: tuple[Span, ...]
    claims: tuple[WorkClaim, ...]
    events: EventLogSnapshot
    diagnostics: ClaimDiagnostics


@dataclass(frozen=True)
class ClaimLedgerCheckpoint:
    """Complete state required to restore without ID/token reuse."""

    ledger_id: str
    domain: ManagedDomain
    claims: tuple[WorkClaim, ...]
    requests: tuple[_ClaimRequest, ...]
    next_claim_id: int
    next_fencing_token: int
    last_observed_at: int
    events: EventLogCheckpoint


def _new_free(domain: ManagedDomain, claims: tuple[WorkClaim, ...] = ()) -> RangeSet:
    free = RangeSet(
        BackendAdapter(IntervalManager()),
        domain=domain,
        initially_available=True,
    )
    for claim in claims:
        if claim.state in {ClaimState.ACTIVE, ClaimState.COMPLETED}:
            mutation = free.discard(claim.span, require_covered=True)
            if not mutation.fully_covered:
                raise ClaimInvariantError(
                    "consuming claims overlap or leave the domain"
                )
    return free


class ClaimLedger:
    """Thread-safe earliest-fit work ledger with process-local fencing tokens."""

    def __init__(self, domain: DomainInput, *, clock: Clock):
        if not hasattr(clock, "now") or not callable(clock.now):
            raise TypeError("clock must provide a callable now()")
        managed = domain if isinstance(domain, ManagedDomain) else ManagedDomain(domain)
        observed_at = validate_coordinate(clock.now(), "clock.now()")
        self._clock = clock
        self._domain = managed
        self._ledger_id = secrets.token_hex(16)
        self._claims: dict[int, WorkClaim] = {}
        self._requests: dict[str, _ClaimRequest] = {}
        self._next_claim_id = 1
        self._next_fencing_token = 1
        self._last_observed_at = observed_at
        self._free = _new_free(managed)
        self._events = EventLog(clock=clock)
        self._lock = RLock()

    @property
    def ledger_id(self) -> str:
        return self._ledger_id

    @property
    def domain(self) -> ManagedDomain:
        return self._domain

    def _now(self) -> int:
        observed = validate_coordinate(self._clock.now(), "clock.now()")
        if observed < self._last_observed_at:
            raise ValueError("clock moved backwards")
        self._last_observed_at = observed
        return observed

    @staticmethod
    def _append_event(
        event_log: EventLog, claim: WorkClaim, occurred_at: int
    ) -> None:
        stream = f"claim:{claim.claim_id}"
        event_log.append(
            stream,
            claim.state.value,
            {
                "ledger_id": claim.ledger_id,
                "claim_id": claim.claim_id,
                "owner": claim.owner,
                "start": claim.span.start,
                "end": claim.span.end,
                "fencing_token": claim.fencing_token,
                "revision": claim.revision,
                "acquired_at": claim.acquired_at,
                "expires_at": claim.expires_at,
                "request_id": claim.request_id,
                "result": dict(claim.result),
            },
            expected_version=claim.revision - 1,
            idempotency_key=f"{claim.revision}:{claim.state.value}",
            occurred_at=occurred_at,
        )

    def _due_transitions(self, now: int) -> tuple[WorkClaim, ...]:
        return tuple(
            replace(current, state=ClaimState.EXPIRED, revision=current.revision + 1)
            for claim_id in sorted(self._claims)
            if (current := self._claims[claim_id]).state is ClaimState.ACTIVE
            and current.expires_at is not None
            and current.expires_at <= now
        )

    def _stage_transitions(
        self, transitions: tuple[WorkClaim, ...], occurred_at: int
    ) -> tuple[dict[int, WorkClaim], RangeSet, EventLog]:
        staged_claims = self._claims.copy()
        for transition in transitions:
            staged_claims[transition.claim_id] = transition
        staged_free = _new_free(self._domain, tuple(staged_claims.values()))
        staged_events = EventLog.from_checkpoint(
            self._events.checkpoint(), clock=self._clock
        )
        for transition in transitions:
            self._append_event(staged_events, transition, occurred_at)
        return staged_claims, staged_free, staged_events

    def _commit_transitions(
        self, staged: tuple[dict[int, WorkClaim], RangeSet, EventLog]
    ) -> None:
        self._claims, self._free, self._events = staged

    def _expire_due(self, now: int) -> tuple[WorkClaim, ...]:
        expired = self._due_transitions(now)
        if expired:
            self._commit_transitions(self._stage_transitions(expired, now))
        return expired

    def _active_handle(self, claim: WorkClaim, owner: str | None) -> WorkClaim:
        """Validate supplied identity and revision before any expiry mutation."""
        if not isinstance(claim, WorkClaim):
            raise TypeError("claim must be a WorkClaim")
        if claim.ledger_id != self._ledger_id:
            raise ForeignClaimError("claim belongs to another ledger")
        expected_owner = claim.owner if owner is None else _nonempty(owner, "owner")
        if expected_owner != claim.owner:
            raise ForeignClaimError("claim belongs to another owner")
        current = self._claims.get(claim.claim_id)
        if current is None:
            raise InvalidClaimError("claim ID is unknown")
        if current.owner != expected_owner:
            raise ForeignClaimError("claim belongs to another owner")
        if current.state is ClaimState.EXPIRED:
            raise ExpiredClaimError("claim has expired")
        if current != claim:
            raise StaleClaimError("claim handle is an old revision")
        if current.state is not ClaimState.ACTIVE:
            raise TerminalClaimError(f"claim is already {current.state.value}")
        return current

    def _stage_lifecycle(
        self, current: WorkClaim, transition: WorkClaim, now: int
    ) -> tuple[dict[int, WorkClaim], RangeSet, EventLog]:
        due = self._due_transitions(now)
        if any(item.claim_id == current.claim_id for item in due):
            self._commit_transitions(self._stage_transitions(due, now))
            raise ExpiredClaimError("claim has expired")
        return self._stage_transitions((*due, transition), now)

    def claim_next(
        self,
        owner: str,
        length: int,
        *,
        not_before: int | None = None,
        not_after: int | None = None,
        ttl: int | None = None,
        request_id: str | None = None,
    ) -> WorkClaim:
        """Claim the lexicographically earliest available contiguous range."""
        owner = _nonempty(owner, "owner")
        validate_length(length)
        start_bound = self._domain.bounds[0] if not_before is None else not_before
        validate_coordinate(start_bound, "not_before")
        if not_after is not None:
            validate_coordinate(not_after, "not_after")
            if not_after <= start_bound:
                raise ValueError("not_after must be greater than not_before")
        validated_ttl = _positive_or_none(ttl, "ttl")
        if request_id is not None:
            request_id = _nonempty(request_id, "request_id")

        with self._lock:
            if request_id is not None:
                prior = self._requests.get(request_id)
                if prior is not None:
                    signature = (
                        owner,
                        length,
                        start_bound,
                        not_after,
                        validated_ttl,
                    )
                    if signature != (
                        prior.owner,
                        prior.length,
                        prior.not_before,
                        prior.not_after,
                        prior.ttl,
                    ):
                        raise ClaimRequestConflictError(
                            "request ID was reused for a different claim"
                        )
                    return self._claims[prior.claim_id]

            now = self._now()
            due = self._due_transitions(now)
            claims_after_expiry = self._claims.copy()
            claims_after_expiry.update(
                (terminal.claim_id, terminal) for terminal in due
            )
            staged_available = _new_free(
                self._domain, tuple(claims_after_expiry.values())
            )
            allocated = staged_available.allocate(
                length,
                not_before=start_bound,
                not_after=not_after,
            )
            if allocated is None:
                if due:
                    self._commit_transitions(self._stage_transitions(due, now))
                raise ClaimUnavailableError("no contiguous work range is available")
            claim = WorkClaim(
                ledger_id=self._ledger_id,
                claim_id=self._next_claim_id,
                owner=owner,
                span=allocated.span,
                fencing_token=self._next_fencing_token,
                acquired_at=now,
                expires_at=(None if validated_ttl is None else now + validated_ttl),
                request_id=request_id,
            )
            request = (
                None
                if request_id is None
                else _ClaimRequest(
                    request_id,
                    owner,
                    length,
                    start_bound,
                    not_after,
                    validated_ttl,
                    claim.claim_id,
                )
            )
            staged = self._stage_transitions((*due, claim), now)
            staged_requests = self._requests.copy()
            if request is not None:
                staged_requests[request.request_id] = request
            self._commit_transitions(staged)
            self._requests = staged_requests
            self._next_claim_id += 1
            self._next_fencing_token += 1
            return claim

    def renew(
        self, claim: WorkClaim, *, ttl: int, owner: str | None = None
    ) -> WorkClaim:
        """Renew active ownership and issue a newer global fencing token."""
        validated_ttl = _positive(ttl, "ttl")
        with self._lock:
            current = self._active_handle(claim, owner)
            now = self._now()
            renewed = replace(
                current,
                fencing_token=self._next_fencing_token,
                expires_at=now + validated_ttl,
                revision=current.revision + 1,
            )
            staged = self._stage_lifecycle(current, renewed, now)
            self._commit_transitions(staged)
            self._next_fencing_token += 1
            return renewed

    def complete(
        self,
        claim: WorkClaim,
        *,
        owner: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> WorkClaim:
        """Record completion metadata without executing application work."""
        frozen_result = freeze_metadata(result)
        with self._lock:
            current = self._active_handle(claim, owner)
            now = self._now()
            completed = replace(
                current,
                state=ClaimState.COMPLETED,
                revision=current.revision + 1,
                result=frozen_result,
            )
            staged = self._stage_lifecycle(current, completed, now)
            self._commit_transitions(staged)
            return completed

    def abandon(self, claim: WorkClaim, *, owner: str | None = None) -> WorkClaim:
        """Return unfinished work to availability and retain terminal evidence."""
        with self._lock:
            current = self._active_handle(claim, owner)
            now = self._now()
            abandoned = replace(
                current,
                state=ClaimState.ABANDONED,
                revision=current.revision + 1,
            )
            staged = self._stage_lifecycle(current, abandoned, now)
            self._commit_transitions(staged)
            return abandoned

    def expire(self) -> tuple[WorkClaim, ...]:
        """Expire every due active claim in claim-ID order."""
        with self._lock:
            return self._expire_due(self._now())

    def events(self) -> EventLogSnapshot:
        """Return immutable transition history."""
        return self._events.snapshot()

    def _diagnostics(self, now: int) -> ClaimDiagnostics:
        states = {state: 0 for state in ClaimState}
        for claim in self._claims.values():
            states[claim.state] += 1
        available = self._free.snapshot().total_free
        return ClaimDiagnostics(
            observed_at=now,
            total_work=self._domain.measure,
            available_work=available,
            active_claims=states[ClaimState.ACTIVE],
            completed_claims=states[ClaimState.COMPLETED],
            abandoned_claims=states[ClaimState.ABANDONED],
            expired_claims=states[ClaimState.EXPIRED],
            issued_claim_ids=self._next_claim_id - 1,
            issued_fencing_tokens=self._next_fencing_token - 1,
            next_claim_id=self._next_claim_id,
            next_fencing_token=self._next_fencing_token,
        )

    def snapshot(self) -> ClaimLedgerSnapshot:
        """Expire due claims and return deterministic detached state."""
        with self._lock:
            now = self._now()
            self._expire_due(now)
            available = tuple(item.span for item in self._free.intervals())
            return ClaimLedgerSnapshot(
                now,
                self._ledger_id,
                self._domain,
                available,
                tuple(self._claims[key] for key in sorted(self._claims)),
                self._events.snapshot(),
                self._diagnostics(now),
            )

    def checkpoint(self) -> ClaimLedgerCheckpoint:
        """Expire due claims and capture all IDs, retries, and event history."""
        with self._lock:
            snapshot = self.snapshot()
            return ClaimLedgerCheckpoint(
                self._ledger_id,
                self._domain,
                snapshot.claims,
                tuple(self._requests[key] for key in sorted(self._requests)),
                self._next_claim_id,
                self._next_fencing_token,
                self._last_observed_at,
                self._events.checkpoint(),
            )

    def invariant_violations(self) -> tuple[str, ...]:
        """Return deterministic structural/accounting violations."""
        with self._lock:
            violations: list[str] = []
            consuming = sorted(
                claim.span
                for claim in self._claims.values()
                if claim.state in {ClaimState.ACTIVE, ClaimState.COMPLETED}
            )
            for left, right in zip(consuming, consuming[1:], strict=False):
                if left.overlaps(right):
                    violations.append("consuming claims overlap")
                    break
            consumed = sum(span.length for span in consuming)
            available = self._free.snapshot().total_free
            if available + consumed != self._domain.measure:
                violations.append("available and consumed work do not partition domain")
            try:
                expected_free = _new_free(self._domain, tuple(self._claims.values()))
            except (ClaimInvariantError, ValueError):
                violations.append("consuming claims cannot reconstruct availability")
            else:
                observed_ranges = tuple(item.span for item in self._free.intervals())
                expected_ranges = tuple(item.span for item in expected_free.intervals())
                if observed_ranges != expected_ranges:
                    violations.append("available geometry does not match claims")
            if any(
                not self._domain.contains(claim.span) for claim in self._claims.values()
            ):
                violations.append("claim is outside the managed domain")
            if self._next_claim_id != max(self._claims, default=0) + 1:
                violations.append("next_claim_id is inconsistent with issued IDs")
            fencing_tokens = [
                claim.fencing_token for claim in self._claims.values()
            ]
            if len(fencing_tokens) != len(set(fencing_tokens)):
                violations.append("claims reuse a fencing token")
            max_token = max(fencing_tokens, default=0)
            if self._next_fencing_token != max_token + 1:
                violations.append(
                    "next_fencing_token is inconsistent with issued tokens"
                )
            for request in self._requests.values():
                claim = self._claims.get(request.claim_id)
                if claim is None or claim.request_id != request.request_id:
                    violations.append("request mapping does not identify its claim")
                    break
            return tuple(violations)

    def validate(self) -> None:
        """Raise with complete invariant evidence when state is invalid."""
        violations = self.invariant_violations()
        if violations:
            raise ClaimInvariantError("; ".join(violations))

    @classmethod
    def from_checkpoint(
        cls, checkpoint: ClaimLedgerCheckpoint, *, clock: Clock
    ) -> ClaimLedger:
        """Validate and restore a single process-local ledger lineage."""
        if not isinstance(checkpoint, ClaimLedgerCheckpoint):
            raise TypeError("checkpoint must be a ClaimLedgerCheckpoint")
        _nonempty(checkpoint.ledger_id, "ledger_id")
        if not isinstance(checkpoint.domain, ManagedDomain):
            raise TypeError("checkpoint domain must be a ManagedDomain")
        validate_coordinate(checkpoint.next_claim_id, "next_claim_id")
        validate_coordinate(checkpoint.next_fencing_token, "next_fencing_token")
        validate_coordinate(checkpoint.last_observed_at, "last_observed_at")
        if checkpoint.next_claim_id <= 0 or checkpoint.next_fencing_token <= 0:
            raise ClaimInvariantError("next counters must be positive")

        observed_clock = validate_coordinate(clock.now(), "clock.now()")
        if observed_clock < checkpoint.last_observed_at:
            raise ClaimInvariantError("clock predates checkpoint")

        claims: dict[int, WorkClaim] = {}
        for claim in checkpoint.claims:
            if not isinstance(claim, WorkClaim):
                raise TypeError("checkpoint claims must be WorkClaim values")
            if claim.ledger_id != checkpoint.ledger_id:
                raise ClaimInvariantError("claim belongs to another ledger")
            if claim.claim_id in claims:
                raise ClaimInvariantError("duplicate claim ID")
            if not checkpoint.domain.contains(claim.span):
                raise ClaimInvariantError("claim is outside the managed domain")
            claims[claim.claim_id] = claim
        requests: dict[str, _ClaimRequest] = {}
        for request in checkpoint.requests:
            if not isinstance(request, _ClaimRequest):
                raise TypeError("checkpoint requests are invalid")
            if request.request_id in requests:
                raise ClaimInvariantError("duplicate claim request ID")
            mapped_claim = claims.get(request.claim_id)
            if mapped_claim is None or mapped_claim.request_id != request.request_id:
                raise ClaimInvariantError("claim request does not match a claim")
            if (
                mapped_claim.owner != request.owner
                or mapped_claim.span.length != request.length
                or mapped_claim.span.start < request.not_before
                or (
                    request.not_after is not None
                    and mapped_claim.span.end > request.not_after
                )
            ):
                raise ClaimInvariantError("claim request arguments do not match claim")
            requests[request.request_id] = request

        expected_request_ids = {
            claim.request_id
            for claim in claims.values()
            if claim.request_id is not None
        }
        if set(requests) != expected_request_ids:
            raise ClaimInvariantError("claim request mappings are incomplete")

        restored_events = EventLog.from_checkpoint(checkpoint.events, clock=clock)
        all_events = restored_events.events()
        events_by_stream: dict[str, list[Event]] = {}
        previous_occurred_at: int | None = None
        for event in all_events:
            events_by_stream.setdefault(event.stream, []).append(event)
            if event.occurred_at > checkpoint.last_observed_at:
                raise ClaimInvariantError("claim event occurs after checkpoint time")
            if (
                previous_occurred_at is not None
                and event.occurred_at < previous_occurred_at
            ):
                raise ClaimInvariantError(
                    "claim event timestamps must be globally monotonic"
                )
            previous_occurred_at = event.occurred_at

        expected_streams = {f"claim:{claim_id}" for claim_id in claims}
        if set(events_by_stream) != expected_streams:
            raise ClaimInvariantError("claim event streams do not match claims")

        payload_fields = {
            "ledger_id",
            "claim_id",
            "owner",
            "start",
            "end",
            "fencing_token",
            "revision",
            "acquired_at",
            "expires_at",
            "request_id",
            "result",
        }
        empty_result = dict(freeze_metadata({"result": {}}))["result"]
        requests_by_sequence = {
            request.event_sequence: request for request in checkpoint.events.requests
        }
        event_payloads: dict[int, dict[str, Any]] = {}
        for claim_id, claim in claims.items():
            stream_events = events_by_stream[f"claim:{claim_id}"]
            if len(stream_events) != claim.revision:
                raise ClaimInvariantError("claim event count does not match revision")

            previous_event: Event | None = None
            previous_payload: dict[str, Any] | None = None
            for revision, event in enumerate(stream_events, start=1):
                payload = dict(event.payload)
                event_payloads[event.sequence] = payload
                if set(payload) != payload_fields:
                    raise ClaimInvariantError("claim event payload fields are invalid")
                if event.idempotency_key != f"{revision}:{event.kind}":
                    raise ClaimInvariantError("claim event idempotency key is invalid")
                event_request = requests_by_sequence[event.sequence]
                if (
                    event_request.expected_version != revision - 1
                    or event_request.occurred_at != event.occurred_at
                ):
                    raise ClaimInvariantError(
                        "claim event idempotency request is inconsistent"
                    )
                if event.version != revision or payload["revision"] != revision:
                    raise ClaimInvariantError("claim event revision is inconsistent")
                if (
                    payload["ledger_id"] != checkpoint.ledger_id
                    or payload["claim_id"] != claim_id
                    or payload["owner"] != claim.owner
                    or payload["start"] != claim.span.start
                    or payload["end"] != claim.span.end
                    or payload["acquired_at"] != claim.acquired_at
                    or payload["request_id"] != claim.request_id
                ):
                    raise ClaimInvariantError("claim event identity is inconsistent")
                integer_fields = (
                    "claim_id",
                    "start",
                    "end",
                    "fencing_token",
                    "revision",
                    "acquired_at",
                )
                if any(type(payload[field]) is not int for field in integer_fields):
                    raise ClaimInvariantError("claim event coordinates are invalid")
                expires_at = payload["expires_at"]
                if expires_at is not None and type(expires_at) is not int:
                    raise ClaimInvariantError("claim event expiry is invalid")

                if revision == 1:
                    if event.kind != ClaimState.ACTIVE.value:
                        raise ClaimInvariantError(
                            "claim history must begin with acquisition"
                        )
                    if event.occurred_at != payload["acquired_at"]:
                        raise ClaimInvariantError(
                            "claim acquisition timestamp is inconsistent"
                        )
                    if expires_at is not None and expires_at <= event.occurred_at:
                        raise ClaimInvariantError("claim acquisition expiry is invalid")
                    if payload["result"] != empty_result:
                        raise ClaimInvariantError(
                            "active claim event cannot contain a result"
                        )
                else:
                    if previous_event is None or previous_payload is None:
                        raise RuntimeError("claim history validation lost prior event")
                    if previous_event.kind != ClaimState.ACTIVE.value:
                        raise ClaimInvariantError(
                            "claim history continues after a terminal transition"
                        )
                    if event.occurred_at < previous_event.occurred_at:
                        raise ClaimInvariantError(
                            "claim event timestamps must be monotonic"
                        )
                    prior_expiry = previous_payload["expires_at"]
                    if event.kind == ClaimState.ACTIVE.value:
                        if payload["fencing_token"] == previous_payload["fencing_token"]:
                            raise ClaimInvariantError(
                                "claim renewal must issue a fencing token"
                            )
                        if expires_at is None or expires_at <= event.occurred_at:
                            raise ClaimInvariantError("claim renewal expiry is invalid")
                        if (
                            prior_expiry is not None
                            and event.occurred_at >= prior_expiry
                        ):
                            raise ClaimInvariantError(
                                "claim renewal occurs after expiry"
                            )
                        if payload["result"] != empty_result:
                            raise ClaimInvariantError(
                                "active claim event cannot contain a result"
                            )
                    elif event.kind in {
                        ClaimState.COMPLETED.value,
                        ClaimState.ABANDONED.value,
                    }:
                        if (
                            payload["fencing_token"]
                            != previous_payload["fencing_token"]
                            or expires_at != prior_expiry
                        ):
                            raise ClaimInvariantError(
                                "terminal claim event changes lease evidence"
                            )
                        if (
                            prior_expiry is not None
                            and event.occurred_at >= prior_expiry
                        ):
                            raise ClaimInvariantError(
                                "terminal claim transition occurs after expiry"
                            )
                        if (
                            event.kind == ClaimState.ABANDONED.value
                            and payload["result"] != empty_result
                        ):
                            raise ClaimInvariantError(
                                "abandoned claim event cannot contain a result"
                            )
                    elif event.kind == ClaimState.EXPIRED.value:
                        if (
                            payload["fencing_token"]
                            != previous_payload["fencing_token"]
                            or expires_at != prior_expiry
                        ):
                            raise ClaimInvariantError(
                                "expired claim event changes lease evidence"
                            )
                        if prior_expiry is None or event.occurred_at < prior_expiry:
                            raise ClaimInvariantError(
                                "claim expiration occurs before lease expiry"
                            )
                        if payload["result"] != empty_result:
                            raise ClaimInvariantError(
                                "expired claim event cannot contain a result"
                            )
                    else:
                        raise ClaimInvariantError("claim event state is invalid")
                previous_event = event
                previous_payload = payload

            first_payload = event_payloads[stream_events[0].sequence]
            final_event = stream_events[-1]
            final_payload = event_payloads[final_event.sequence]
            expected_result = dict(
                freeze_metadata({"result": dict(claim.result)})
            )["result"]
            if (
                final_event.kind != claim.state.value
                or final_payload["revision"] != claim.revision
                or final_payload["fencing_token"] != claim.fencing_token
                or final_payload["expires_at"] != claim.expires_at
                or final_payload["result"] != expected_result
            ):
                raise ClaimInvariantError("final claim event does not match claim")
            if (
                claim.state is ClaimState.ACTIVE
                and claim.expires_at is not None
                and claim.expires_at <= checkpoint.last_observed_at
            ):
                raise ClaimInvariantError("checkpoint retains an expired active claim")
            if claim.request_id is not None:
                request = requests[claim.request_id]
                expected_initial_expiry = (
                    None
                    if request.ttl is None
                    else claim.acquired_at + request.ttl
                )
                if first_payload["expires_at"] != expected_initial_expiry:
                    raise ClaimInvariantError(
                        "claim request TTL does not match acquisition event"
                    )

        next_claim_id = 1
        next_fencing_token = 1
        current_tokens: dict[str, int] = {}
        for event in all_events:
            payload = event_payloads[event.sequence]
            if event.version == 1:
                if payload["claim_id"] != next_claim_id:
                    raise ClaimInvariantError("claim IDs are issued out of order")
                next_claim_id += 1
            issues_token = event.version == 1 or event.kind == ClaimState.ACTIVE.value
            if issues_token:
                if payload["fencing_token"] != next_fencing_token:
                    raise ClaimInvariantError(
                        "fencing tokens are duplicate or issued out of order"
                    )
                current_tokens[event.stream] = next_fencing_token
                next_fencing_token += 1
            elif payload["fencing_token"] != current_tokens.get(event.stream):
                raise ClaimInvariantError(
                    "terminal event has an inconsistent fencing token"
                )
        if checkpoint.next_claim_id != next_claim_id:
            raise ClaimInvariantError(
                "next_claim_id would reuse or skip claim IDs"
            )
        if checkpoint.next_fencing_token != next_fencing_token:
            raise ClaimInvariantError(
                "next_fencing_token would reuse or skip fencing tokens"
            )

        candidate = cls(checkpoint.domain, clock=clock)
        candidate._ledger_id = checkpoint.ledger_id
        candidate._claims = claims
        candidate._requests = requests
        candidate._next_claim_id = checkpoint.next_claim_id
        candidate._next_fencing_token = checkpoint.next_fencing_token
        candidate._last_observed_at = checkpoint.last_observed_at
        candidate._free = _new_free(checkpoint.domain, checkpoint.claims)
        candidate._events = restored_events
        candidate._now()
        candidate.validate()
        return candidate
