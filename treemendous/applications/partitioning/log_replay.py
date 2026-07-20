"""Idempotent offset-event replay with deterministic state checkpoints."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

from treemendous.applications._shared.claiming import ClaimUnavailableError, WorkClaim
from treemendous.applications._shared.clock import Clock
from treemendous.applications.partitioning._runtime import (
    PartitionRuntime,
    nonempty,
    positive,
)

ReplayValue: TypeAlias = int | str


@dataclass(frozen=True, order=True)
class ReplayEvent:
    """One uniquely offset mutation: set, increment, or delete."""

    offset: int
    key: str
    operation: str
    value: ReplayValue | None = None


@dataclass(frozen=True)
class LogReplayAuditSnapshot:
    """Immutable, non-restorable materialized state and applied offsets."""

    state: tuple[tuple[str, ReplayValue], ...]
    applied_offsets: tuple[int, ...]
    runtime: object


class LogReplayEngine:
    """Apply claimed offset events idempotently in global offset order.

    Rebuilding from the applied-offset set makes out-of-order worker completion
    deterministic. The state is process-local; durable distributed replay must
    transactionally persist offsets, values, and fencing tokens.
    """

    def __init__(
        self, events: object, *, clock: Clock | None = None
    ) -> None:
        if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
            raise TypeError("events must be a sequence")
        if not events:
            raise ValueError("events must not be empty")
        checked: list[ReplayEvent] = []
        seen: set[int] = set()
        for event in cast(Sequence[object], events):
            if not isinstance(event, ReplayEvent):
                raise TypeError("events must contain ReplayEvent values")
            if type(event.offset) is not int or event.offset < 0:
                raise ValueError("event offsets must be non-negative integers")
            if event.offset in seen:
                raise ValueError("event offsets must be unique")
            seen.add(event.offset)
            nonempty(event.key, "event key")
            if event.operation not in {"set", "increment", "delete"}:
                raise ValueError("event operation must be set, increment, or delete")
            if event.operation == "set" and not isinstance(event.value, (int, str)):
                raise TypeError("set events require an integer or string value")
            if event.operation == "increment" and type(event.value) is not int:
                raise TypeError("increment events require an integer value")
            if event.operation == "delete" and event.value is not None:
                raise ValueError("delete events cannot have a value")
            checked.append(event)
        self._events = tuple(sorted(checked))
        self._by_offset = {event.offset: event for event in self._events}
        self._applied: set[int] = set()
        self._state: dict[str, ReplayValue] = {}
        self._runtime = PartitionRuntime(len(self._events), clock=clock)

    def claim(self, owner: str, length: int) -> WorkClaim:
        """Claim event positions in sorted offset order."""
        return self._runtime.claim(owner, length)

    def _materialize(self, applied: set[int]) -> dict[str, ReplayValue]:
        state: dict[str, ReplayValue] = {}
        for offset in sorted(applied):
            event = self._by_offset[offset]
            if event.operation == "delete":
                state.pop(event.key, None)
            elif event.operation == "set":
                if event.value is None:
                    raise RuntimeError("validated set event lost its value")
                state[event.key] = event.value
            else:
                current = state.get(event.key, 0)
                if type(current) is not int:
                    raise ValueError("increment cannot be applied to a string value")
                if not isinstance(event.value, int):
                    raise RuntimeError("validated increment event lost its value")
                state[event.key] = current + event.value
        return state

    def apply_claim(self, claim: WorkClaim) -> tuple[int, ...]:
        """Apply a band once and materialize state by ascending offset."""
        def prepare() -> tuple[tuple[int, ...], set[int], dict[str, ReplayValue]]:
            offsets = tuple(
                self._events[position].offset
                for position in range(claim.span.start, claim.span.end)
            )
            applied = self._applied | set(offsets)
            return offsets, applied, self._materialize(applied)

        def commit(
            value: tuple[tuple[int, ...], set[int], dict[str, ReplayValue]],
        ) -> None:
            self._applied, self._state = value[1], value[2]

        prepared = self._runtime.execute_claim(
            claim,
            kind="replayed",
            prepare=prepare,
            commit=commit,
            result=lambda value: {"events": len(value[0])},
        )
        return prepared[0]

    def run(self, *, window_size: int = 128) -> tuple[tuple[str, ReplayValue], ...]:
        """Replay all remaining offsets and return sorted materialized state."""
        positive(window_size, "window_size")
        while True:
            try:
                claim = self.claim("local", window_size)
            except ClaimUnavailableError:
                break
            self.apply_claim(claim)
        return tuple(sorted(self._state.items()))

    def _snapshot(self) -> LogReplayAuditSnapshot:
        return LogReplayAuditSnapshot(
            tuple(sorted(self._state.items())),
            tuple(sorted(self._applied)),
            self._runtime.checkpoint(),
        )

    def snapshot(self) -> LogReplayAuditSnapshot:
        """Capture non-restorable materialized and idempotency evidence."""
        return self._runtime.observe(self._snapshot)

    audit_snapshot = snapshot


def create_log_replay(
    events: Sequence[ReplayEvent] = (
        ReplayEvent(0, "count", "set", 1),
        ReplayEvent(1, "count", "increment", 2),
    ),
    *,
    clock: Clock | None = None,
) -> LogReplayEngine:
    """Create an idempotent offset replay job."""
    return LogReplayEngine(events, clock=clock)
