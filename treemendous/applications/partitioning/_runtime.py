"""Private process-local runtime shared by partitioning application engines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any, TypeVar

from treemendous.applications._shared.claiming import (
    ClaimInvariantError,
    ClaimLedger,
    ClaimLedgerCheckpoint,
    ClaimState,
    StaleClaimError,
    WorkClaim,
)
from treemendous.applications._shared.clock import Clock, LogicalClock
from treemendous.applications._shared.events import EventLog, EventLogCheckpoint
from treemendous.domain import validate_coordinate, validate_length

_ResultT = TypeVar("_ResultT")
_SnapshotT = TypeVar("_SnapshotT")


@dataclass(frozen=True)
class RuntimeCheckpoint:
    """Private, restorable checkpoint of claims and application audit events."""

    claims: ClaimLedgerCheckpoint
    events: EventLogCheckpoint


class PartitionRuntime:
    """Claim finite ordinal work and atomically fence application commits.

    This helper is deliberately not a distributed coordinator. Checkpoints must
    be durably stored by a caller, and a real multi-process deployment must
    enforce the returned fencing tokens at its result store.
    """

    def __init__(self, work_items: int, *, clock: Clock | None = None) -> None:
        validate_length(work_items)
        self.clock = LogicalClock() if clock is None else clock
        self.ledger = ClaimLedger((0, work_items), clock=self.clock)
        self.log = EventLog(clock=self.clock)
        self._lock = RLock()
        self._executing_claim_ids: set[int] = set()

    def claim(
        self, owner: str, length: int, *, request_id: str | None = None
    ) -> WorkClaim:
        """Claim the next ordinal band, capped at its earliest free interval."""
        validate_length(length)
        with self._lock:
            snapshot = self.ledger.snapshot()
            if not snapshot.available:
                return self.ledger.claim_next(owner, length, request_id=request_id)
            actual = min(length, snapshot.available[0].length)
            return self.ledger.claim_next(owner, actual, request_id=request_id)

    def validate_active(self, claim: WorkClaim) -> WorkClaim:
        """Reject foreign, stale, expired, or terminal handles without effects."""
        with self._lock:
            return self.ledger.validate_active(claim)

    def _ensure_not_executing(self, claim: WorkClaim) -> None:
        if claim.claim_id in self._executing_claim_ids:
            raise StaleClaimError("claim execution is already in progress")

    def _staged_transition(
        self,
        claim: WorkClaim,
        *,
        kind: str,
        result: dict[str, Any] | None,
        abandon: bool,
    ) -> tuple[ClaimLedger, EventLog, WorkClaim]:
        staged_ledger = ClaimLedger.from_checkpoint(
            self.ledger.checkpoint(), clock=self.clock
        )
        staged_log = EventLog.from_checkpoint(self.log.checkpoint(), clock=self.clock)
        if abandon:
            transitioned = staged_ledger.abandon(claim)
            event_kind = "abandoned"
            payload = {"start": claim.span.start, "end": claim.span.end}
        else:
            transitioned = staged_ledger.complete(claim, result=result)
            event_kind = kind
            payload = {
                "start": claim.span.start,
                "end": claim.span.end,
                "fencing_token": claim.fencing_token,
            }
        staged_log.append(
            f"work:{claim.claim_id}",
            event_kind,
            payload,
            expected_version=0,
            idempotency_key=event_kind,
        )
        return staged_ledger, staged_log, transitioned

    def execute_claim(
        self,
        claim: WorkClaim,
        *,
        kind: str,
        prepare: Callable[[], _ResultT],
        commit: Callable[[_ResultT], None],
        result: Callable[[_ResultT], dict[str, Any] | None],
        abandon_on_error: bool = True,
    ) -> _ResultT:
        """Validate, execute, and atomically fence one application-state commit.

        Validation occurs while the runtime lock is held and before ``prepare``
        can read application inputs or invoke caller code. The application must
        stage its new state in ``prepare`` and make ``commit`` a no-fail swap.
        Runtime transition/event state is fully staged before that swap, so a
        failed runtime completion cannot be preceded by an application commit.
        """
        with self._lock, self.ledger.active_transaction(claim):
            self._ensure_not_executing(claim)
            self._executing_claim_ids.add(claim.claim_id)
            try:
                try:
                    prepared = prepare()
                    metadata = result(prepared)
                except Exception:
                    if abandon_on_error:
                        staged_ledger, staged_log, _ = self._staged_transition(
                            claim, kind="abandoned", result=None, abandon=True
                        )
                        self.ledger, self.log = staged_ledger, staged_log
                    raise
                staged_ledger, staged_log, _ = self._staged_transition(
                    claim, kind=kind, result=metadata, abandon=False
                )
                commit(prepared)
                self.ledger, self.log = staged_ledger, staged_log
                return prepared
            finally:
                self._executing_claim_ids.remove(claim.claim_id)

    def abandon_claim(
        self, claim: WorkClaim, *, commit: Callable[[], None] | None = None
    ) -> WorkClaim:
        """Validate and atomically abandon a claim with an optional state swap."""
        with self._lock, self.ledger.active_transaction(claim):
            self._ensure_not_executing(claim)
            staged_ledger, staged_log, abandoned = self._staged_transition(
                claim, kind="abandoned", result=None, abandon=True
            )
            if commit is not None:
                commit()
            self.ledger, self.log = staged_ledger, staged_log
            return abandoned

    def complete(
        self, claim: WorkClaim, kind: str, result: dict[str, Any] | None = None
    ) -> WorkClaim:
        """Atomically record completion evidence without application state."""
        with self._lock, self.ledger.active_transaction(claim):
            self._ensure_not_executing(claim)
            staged_ledger, staged_log, completed = self._staged_transition(
                claim, kind=kind, result=result, abandon=False
            )
            self.ledger, self.log = staged_ledger, staged_log
            return completed

    def abandon(self, claim: WorkClaim) -> WorkClaim:
        """Return an unsuccessfully executed band for retry."""
        return self.abandon_claim(claim)

    def observe(self, snapshot: Callable[[], _SnapshotT]) -> _SnapshotT:
        """Observe application state while transitions are excluded."""
        with self._lock:
            return snapshot()

    def audit_snapshot(
        self, snapshot: Callable[[], _SnapshotT]
    ) -> tuple[_SnapshotT, RuntimeCheckpoint]:
        """Capture non-restorable application evidence and consistent runtime state."""
        with self._lock:
            application = snapshot()
            return application, self.checkpoint()

    def checkpoint(self) -> RuntimeCheckpoint:
        """Capture one internally consistent, restorable runtime state."""
        with self._lock:
            checkpoint = RuntimeCheckpoint(
                self.ledger.checkpoint(), self.log.checkpoint()
            )
            self._validate_checkpoint(checkpoint)
            return checkpoint

    @staticmethod
    def _validate_checkpoint(checkpoint: RuntimeCheckpoint) -> None:
        claims = {claim.claim_id: claim for claim in checkpoint.claims.claims}
        work_events = checkpoint.events.events
        terminal = {
            claim.claim_id: claim
            for claim in claims.values()
            if claim.state in {ClaimState.COMPLETED, ClaimState.ABANDONED}
        }
        if len(work_events) != len(terminal):
            raise ClaimInvariantError("runtime events do not match terminal claims")

        event_claim_ids: list[int] = []
        for expected_sequence, event in enumerate(work_events, start=1):
            if event.sequence != expected_sequence:
                raise ClaimInvariantError(
                    "runtime event transition IDs must be ordered and unique"
                )
            if not event.stream.startswith("work:"):
                raise ClaimInvariantError("runtime event stream is invalid")
            try:
                claim_id = int(event.stream.removeprefix("work:"))
            except ValueError as exc:
                raise ClaimInvariantError("runtime event claim ID is invalid") from exc
            if claim_id in event_claim_ids:
                raise ClaimInvariantError(
                    "runtime events must match terminal claims exactly once"
                )
            if event.version != 1:
                raise ClaimInvariantError(
                    "runtime event transition IDs must be ordered and unique"
                )
            event_claim_ids.append(claim_id)
            claim = terminal.get(claim_id)
            if claim is None:
                raise ClaimInvariantError("runtime event has no terminal claim")
            payload = dict(event.payload)
            expected_fields = {"start", "end"}
            if claim.state is ClaimState.COMPLETED:
                expected_fields.add("fencing_token")
            if set(payload) != expected_fields or (
                payload.get("start") != claim.span.start
                or payload.get("end") != claim.span.end
            ):
                raise ClaimInvariantError("runtime event span contradicts its claim")
            if claim.state is ClaimState.ABANDONED:
                if event.kind != "abandoned":
                    raise ClaimInvariantError(
                        "runtime abandonment event is inconsistent"
                    )
            elif payload.get("fencing_token") != claim.fencing_token:
                raise ClaimInvariantError("runtime completion token is inconsistent")
            if event.idempotency_key != event.kind:
                raise ClaimInvariantError("runtime event transition ID is invalid")

        if set(event_claim_ids) != set(terminal):
            raise ClaimInvariantError(
                "runtime events must match terminal claims exactly once"
            )
        terminal_transition_ids = tuple(
            dict(event.payload).get("claim_id")
            for event in checkpoint.claims.events.events
            if event.kind in {ClaimState.COMPLETED.value, ClaimState.ABANDONED.value}
        )
        if tuple(event_claim_ids) != terminal_transition_ids:
            raise ClaimInvariantError(
                "runtime event transitions contradict terminal claim order"
            )

    @classmethod
    def from_checkpoint(
        cls, checkpoint: RuntimeCheckpoint, *, clock: Clock
    ) -> PartitionRuntime:
        """Restore a runtime lineage after validating cross-kernel evidence."""
        if not isinstance(checkpoint, RuntimeCheckpoint):
            raise TypeError("checkpoint must be a RuntimeCheckpoint")
        cls._validate_checkpoint(checkpoint)
        candidate = cls.__new__(cls)
        candidate.clock = clock
        candidate.ledger = ClaimLedger.from_checkpoint(checkpoint.claims, clock=clock)
        candidate.log = EventLog.from_checkpoint(checkpoint.events, clock=clock)
        candidate._lock = RLock()
        candidate._executing_claim_ids = set()
        return candidate


def positive(value: int, name: str) -> int:
    """Validate a positive integer without accepting booleans."""
    validate_coordinate(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def nonempty(value: str, name: str) -> str:
    """Validate a nonempty string."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value
