"""Log-replay engine contracts."""

import pytest

from tests.oracles.applications.partitioning.log_replay import expected_state
from treemendous.applications.partitioning.log_replay import (
    LogReplayEngine,
    ReplayEvent,
)


def test_offset_replay_is_idempotent_and_audited() -> None:
    events = (
        ReplayEvent(4, "x", "increment", 2),
        ReplayEvent(1, "x", "set", 3),
        ReplayEvent(8, "gone", "set", "v"),
        ReplayEvent(9, "gone", "delete"),
    )
    engine = LogReplayEngine(events)
    observed = engine.run(window_size=2)
    raw = tuple((e.offset, e.key, e.operation, e.value) for e in events)
    expected = (("x", 5),)
    expected_offsets = (1, 4, 8, 9)
    assert observed == expected_state(raw) == expected
    assert engine.audit_snapshot().applied_offsets == expected_offsets
    assert engine.run() == observed


def test_log_replay_rolls_back_invalid_increment() -> None:
    engine = LogReplayEngine(
        (ReplayEvent(0, "x", "set", "text"), ReplayEvent(1, "x", "increment", 1))
    )
    with pytest.raises(ValueError, match="string"):
        engine.run(window_size=2)
    empty: tuple[tuple[str, int | str], ...] = ()
    assert engine.snapshot().state == empty
