"""Contracts for deterministic application clocks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from treemendous.applications._shared.clock import LogicalClock


def test_logical_clock_is_monotonic_and_validated() -> None:
    clock = LogicalClock(-2)
    assert clock.now() == -2
    assert clock.advance() == -1
    assert clock.advance(4) == 3
    assert clock.set(10) == 10
    assert clock.set(10) == 10
    with pytest.raises(ValueError, match="backwards"):
        clock.set(9)
    with pytest.raises(ValueError, match="greater than zero"):
        clock.advance(0)
    with pytest.raises(TypeError, match="integer"):
        LogicalClock(True)


def test_logical_clock_serializes_concurrent_advances() -> None:
    clock = LogicalClock()
    with ThreadPoolExecutor(max_workers=8) as executor:
        observed = list(executor.map(lambda _: clock.advance(), range(200)))

    assert sorted(observed) == list(range(1, 201))
    assert clock.now() == 200
