"""Attested benchmark for deterministic offset-log replay."""

from __future__ import annotations

from tests.oracles.applications.partitioning.log_replay import expected_state
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from tests.performance.applications.partitioning._common import validate_case
from treemendous.applications.partitioning.log_replay import (
    LogReplayEngine,
    ReplayEvent,
    ReplayValue,
)

_DEFAULT_OPERATIONS = 320
_MAX_OPERATIONS = 1_500
_DEFAULT_SEED = 41
_WINDOW_SIZE = 23


def run_benchmark(
    operations: int = _DEFAULT_OPERATIONS, seed: int = _DEFAULT_SEED
) -> ApplicationSample:
    """Replay mixed integer mutations and attest state plus idempotency offsets."""
    validate_case(operations, seed, maximum=_MAX_OPERATIONS)
    events = tuple(
        ReplayEvent(
            index,
            f"key-{(index * 5 + seed) % 17:02}",
            (
                "set"
                if index % 19 == 0
                else "delete"
                if index % 13 == 0
                else "increment"
            ),
            None if index % 13 == 0 and index % 19 != 0 else (index % 7) + 1,
        )
        for index in range(operations)
    )
    raw_events = tuple(
        (event.offset, event.key, event.operation, event.value) for event in events
    )
    engine = LogReplayEngine(tuple(reversed(events)))

    def execute() -> tuple[tuple[str, ReplayValue], ...]:
        return engine.run(window_size=_WINDOW_SIZE)

    def observe(
        raw: tuple[tuple[str, ReplayValue], ...],
    ) -> ApplicationOutcome:
        checkpoint = engine.snapshot()
        return ApplicationOutcome(
            results=raw,
            final_state={
                "state": checkpoint.state,
                "applied_offsets": checkpoint.applied_offsets,
            },
            counters={
                "events_applied": len(checkpoint.applied_offsets),
                "state_entries": len(checkpoint.state),
                "replay_windows": (operations + _WINDOW_SIZE - 1) // _WINDOW_SIZE,
                "run_calls": 1,
            },
        )

    def oracle() -> ApplicationOutcome:
        state = expected_state(raw_events)
        return ApplicationOutcome(
            results=state,
            final_state={
                "state": state,
                "applied_offsets": tuple(range(operations)),
            },
            counters={
                "events_applied": operations,
                "state_entries": len(state),
                "replay_windows": (operations + _WINDOW_SIZE - 1) // _WINDOW_SIZE,
                "run_calls": 1,
            },
        )

    return run_application_case(
        scenario_id="distributed-log-replay",
        operations=operations,
        execute=execute,
        observe=observe,
        oracle=oracle,
    )


def run_smoke() -> int:
    """Run the default attested workload and return its operation count."""
    return run_benchmark().operations
