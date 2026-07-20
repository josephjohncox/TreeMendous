"""Correctness-checked smoke workload for offset log replay."""

from tests.oracles.applications.partitioning.log_replay import expected_state
from treemendous.applications.partitioning.log_replay import (
    LogReplayEngine,
    ReplayEvent,
)


def run_smoke() -> int:
    events = tuple(ReplayEvent(i, f"key-{i % 9}", "increment", 1) for i in range(500))
    engine = LogReplayEngine(tuple(reversed(events)))
    observed = engine.run(window_size=23)
    raw = tuple((e.offset, e.key, e.operation, e.value) for e in events)
    if observed != expected_state(raw):
        raise AssertionError("log replay smoke differs from sequential oracle")
    return len(engine.checkpoint().applied_offsets)
