"""Private process-local runtime shared by partitioning application engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from treemendous.applications._shared.claiming import (
    ClaimLedger,
    ClaimLedgerCheckpoint,
    WorkClaim,
)
from treemendous.applications._shared.clock import Clock, LogicalClock
from treemendous.applications._shared.events import EventLog, EventLogCheckpoint
from treemendous.domain import validate_coordinate, validate_length


@dataclass(frozen=True)
class RuntimeCheckpoint:
    """Private checkpoint of claims and application audit events."""

    claims: ClaimLedgerCheckpoint
    events: EventLogCheckpoint


class PartitionRuntime:
    """Claim finite ordinal work and audit results inside one Python process.

    This helper is deliberately not a distributed coordinator. Checkpoints must
    be durably stored by a caller, and a real multi-process deployment must
    enforce the returned fencing tokens at its result store.
    """

    def __init__(self, work_items: int, *, clock: Clock | None = None) -> None:
        validate_length(work_items)
        self.clock = LogicalClock() if clock is None else clock
        self.ledger = ClaimLedger((0, work_items), clock=self.clock)
        self.log = EventLog(clock=self.clock)

    def claim(self, owner: str, length: int, *, request_id: str | None = None) -> WorkClaim:
        """Claim the next ordinal band, capped at its earliest free interval."""
        validate_length(length)
        snapshot = self.ledger.snapshot()
        if not snapshot.available:
            return self.ledger.claim_next(owner, length, request_id=request_id)
        actual = min(length, snapshot.available[0].length)
        return self.ledger.claim_next(owner, actual, request_id=request_id)

    def complete(
        self, claim: WorkClaim, kind: str, result: dict[str, Any] | None = None
    ) -> WorkClaim:
        """Atomically record local completion evidence after application work."""
        completed = self.ledger.complete(claim, result=result)
        self.log.append(
            f"work:{claim.claim_id}",
            kind,
            {
                "start": claim.span.start,
                "end": claim.span.end,
                "fencing_token": claim.fencing_token,
            },
            expected_version=0,
            idempotency_key="completed",
        )
        return completed

    def abandon(self, claim: WorkClaim) -> WorkClaim:
        """Return an unsuccessfully executed band for retry."""
        abandoned = self.ledger.abandon(claim)
        self.log.append(
            f"work:{claim.claim_id}",
            "abandoned",
            {"start": claim.span.start, "end": claim.span.end},
            expected_version=0,
            idempotency_key="abandoned",
        )
        return abandoned

    def checkpoint(self) -> RuntimeCheckpoint:
        """Capture process-local claim and event state."""
        return RuntimeCheckpoint(self.ledger.checkpoint(), self.log.checkpoint())

    @classmethod
    def from_checkpoint(
        cls, checkpoint: RuntimeCheckpoint, *, clock: Clock
    ) -> PartitionRuntime:
        """Restore a runtime lineage after validating both private kernels."""
        if not isinstance(checkpoint, RuntimeCheckpoint):
            raise TypeError("checkpoint must be a RuntimeCheckpoint")
        candidate = cls.__new__(cls)
        candidate.clock = clock
        candidate.ledger = ClaimLedger.from_checkpoint(checkpoint.claims, clock=clock)
        candidate.log = EventLog.from_checkpoint(checkpoint.events, clock=clock)
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
