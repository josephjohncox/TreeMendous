"""Contracts for correctness-first concrete application timing."""

from __future__ import annotations

import math
from collections.abc import Iterator
from uuid import UUID

import pytest

from tests.performance.applications.harness import (
    ApplicationOutcome,
    canonicalize,
    evidence_checksum,
    run_application_case,
)


def _timer(values: list[int], events: list[str]) -> Iterator[int]:
    for value in values:
        events.append("timer")
        yield value


def test_only_execution_is_inside_timing_and_same_state_is_attested() -> None:
    events: list[str] = []
    state: list[int] = []
    ticks = _timer([100, 145], events)

    def execute() -> tuple[int, ...]:
        events.append("execute")
        state.extend((2, 3, 5))
        return tuple(state)

    def observe(raw: tuple[int, ...]) -> ApplicationOutcome:
        events.append("observe")
        return ApplicationOutcome(raw, tuple(state), {"mutations": len(state)})

    def oracle() -> ApplicationOutcome:
        events.append("oracle")
        return ApplicationOutcome((2, 3, 5), (2, 3, 5), {"mutations": 3})

    sample = run_application_case(
        scenario_id="synthetic-application",
        operations=3,
        execute=execute,
        observe=observe,
        oracle=oracle,
        timer=lambda: next(ticks),
    )

    assert events == ["timer", "execute", "timer", "observe", "oracle"]
    assert sample.execution_ns == 45
    assert sample.operations == 3
    assert sample.validated
    expected_state_checksum = evidence_checksum([2, 3, 5])
    assert sample.state_checksum == expected_state_checksum


def test_mismatch_rejects_timing_evidence() -> None:
    with pytest.raises(AssertionError, match="evidence differs"):
        run_application_case(
            scenario_id="mismatch",
            operations=1,
            execute=lambda: 1,
            observe=lambda raw: ApplicationOutcome(raw, (), {"operations": 1}),
            oracle=lambda: ApplicationOutcome(2, (), {"operations": 1}),
            timer=iter((1, 2)).__next__,
        )


def test_canonicalization_is_order_independent_and_rejects_unsafe_values() -> None:
    left = {"values": {3, 1, 2}, "mapping": {2: "b", 1: "a"}}
    right = {"mapping": {1: "a", 2: "b"}, "values": {2, 3, 1}}
    assert canonicalize(left) == canonicalize(right)
    assert evidence_checksum(left) == evidence_checksum(right)
    assert canonicalize(UUID("12345678-1234-5678-1234-567812345678")) == {
        "uuid": "12345678-1234-5678-1234-567812345678"
    }
    with pytest.raises(ValueError, match="finite"):
        canonicalize(math.inf)
    with pytest.raises(TypeError, match="unsupported"):
        canonicalize(object())


@pytest.mark.parametrize("operations", [0, -1])
def test_operation_count_must_be_positive(operations: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        run_application_case(
            scenario_id="invalid-count",
            operations=operations,
            execute=lambda: None,
            observe=lambda raw: ApplicationOutcome(raw, (), {}),
            oracle=lambda: ApplicationOutcome(None, (), {}),
        )
