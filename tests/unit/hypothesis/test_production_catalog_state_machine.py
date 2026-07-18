"""Reference-model state machine for every stable production backend."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import assume, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    rule,
    run_state_machine_as_test,
)

from treemendous.backends import (
    CATALOG,
    Available,
    Invalid,
    Maturity,
    Unavailable,
    probe_backend,
)
from treemendous.backends.adapters import BackendAdapter
from treemendous.domain import IntervalResult, Span
from treemendous.rangeset import RangeSet

DOMAIN_SIZE = 20
STABLE_SPECS = tuple(spec for spec in CATALOG if spec.maturity is Maturity.STABLE)
SPANS = st.tuples(st.integers(0, DOMAIN_SIZE - 1), st.integers(1, DOMAIN_SIZE))


def _runs(bits: list[bool], value: bool = True) -> tuple[Span, ...]:
    result: list[Span] = []
    start: int | None = None
    for index, bit in enumerate([*bits, not value]):
        if bit is value and start is None:
            start = index
        elif bit is not value and start is not None:
            result.append(Span(start, index))
            start = None
    return tuple(result)


def _machine_for(spec: Any):
    class ProductionCatalogMachine(RuleBasedStateMachine):
        def __init__(self) -> None:
            super().__init__()
            implementation = spec.loader()(**dict(spec.constructor_args))
            self.ranges = RangeSet(
                BackendAdapter(implementation),
                domain=(0, DOMAIN_SIZE),
                initially_available=False,
            )
            self.model = [False] * DOMAIN_SIZE

        @rule(bounds=SPANS)
        def add(self, bounds: tuple[int, int]) -> None:
            start, end = bounds
            assume(start < end)
            before = self.model.copy()
            result = self.ranges.add(Span(start, end))
            for point in range(start, end):
                self.model[point] = True
            changed_bits = [
                start <= point < end and not before[point]
                for point in range(DOMAIN_SIZE)
            ]
            assert result.changed == _runs(changed_bits)
            assert result.changed_length == sum(changed_bits)
            assert result.fully_covered == all(before[start:end])

        @rule(bounds=SPANS, require_covered=st.booleans())
        def discard(self, bounds: tuple[int, int], require_covered: bool) -> None:
            start, end = bounds
            assume(start < end)
            before = self.model.copy()
            covered = all(before[start:end])
            result = self.ranges.discard(
                Span(start, end), require_covered=require_covered
            )
            if not require_covered or covered:
                for point in range(start, end):
                    self.model[point] = False
            changed_bits = [
                start <= point < end and before[point] and not self.model[point]
                for point in range(DOMAIN_SIZE)
            ]
            assert result.changed == _runs(changed_bits)
            assert result.changed_length == sum(changed_bits)
            assert result.fully_covered == covered

        def _expected_fit(self, length: int, not_before: int) -> Span | None:
            for start in range(not_before, DOMAIN_SIZE - length + 1):
                if all(self.model[start : start + length]):
                    return Span(start, start + length)
            return None

        @rule(
            length=st.integers(1, DOMAIN_SIZE),
            not_before=st.integers(0, DOMAIN_SIZE - 1),
        )
        def first_fit(self, length: int, not_before: int) -> None:
            result = self.ranges.first_fit(length, not_before=not_before)
            expected = self._expected_fit(length, not_before)
            assert (None if result is None else result.span) == expected

        @rule(
            length=st.integers(1, DOMAIN_SIZE),
            not_before=st.integers(0, DOMAIN_SIZE - 1),
        )
        def allocate(self, length: int, not_before: int) -> None:
            expected = self._expected_fit(length, not_before)
            result = self.ranges.allocate(length, not_before=not_before)
            assert (None if result is None else result.span) == expected
            if expected is not None:
                for point in range(expected.start, expected.end):
                    self.model[point] = False

        @rule()
        def invalid_mutations_are_atomic(self) -> None:
            before = self.ranges.snapshot()
            with pytest.raises(ValueError):
                self.ranges.add(Span(1, 1))
            with pytest.raises(ValueError):
                self.ranges.discard(Span(2, 1))
            assert self.ranges.snapshot() == before

        @invariant()
        def observable_state_matches_model(self) -> None:
            expected_intervals = tuple(
                IntervalResult(span.start, span.end) for span in _runs(self.model)
            )
            snapshot = self.ranges.snapshot()
            assert snapshot.intervals == expected_intervals
            assert snapshot.total_free == sum(self.model)
            stats = self.ranges.stats()
            assert stats.total_free == sum(self.model)
            assert stats.total_occupied == DOMAIN_SIZE - sum(self.model)
            assert stats.free_chunks == len(expected_intervals)
            assert stats.largest_chunk == max(
                (span.length for span in _runs(self.model)), default=0
            )

    return ProductionCatalogMachine


@pytest.mark.parametrize("spec", STABLE_SPECS, ids=lambda spec: spec.id)
def test_stable_production_catalog_state_machine(spec: Any) -> None:
    state = probe_backend(spec)
    if isinstance(state, Unavailable):
        pytest.fail(f"stable backend {spec.id} unavailable: {state.reason}")
    assert not isinstance(state, Invalid), f"{spec.id}: {state.error}"
    assert isinstance(state, Available)
    run_state_machine_as_test(
        _machine_for(spec),
        settings=settings(max_examples=20, stateful_step_count=25, deadline=None),
    )
