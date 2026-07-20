"""Mutation/query/snapshot state machines for optimized box indexes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from tests.oracles.multidimensional.brute_box_index import BruteBoxIndex
from treemendous.multidimensional import (
    BoundedBoxIndex,
    Box,
    BoxHandle,
    BoxIndex2D,
    BoxIndex3D,
    BoxIndex4D,
    BoxIndexProtocol,
)


def _box_bounds(
    dimensions: int,
) -> st.SearchStrategy[tuple[tuple[int, ...], tuple[int, ...]]]:
    lower = st.tuples(
        *(st.integers(min_value=-3, max_value=2) for _ in range(dimensions))
    )
    widths = st.tuples(
        *(st.integers(min_value=1, max_value=2) for _ in range(dimensions))
    )
    return st.tuples(lower, widths).map(
        lambda pair: (
            pair[0],
            tuple(value + width for value, width in zip(pair[0], pair[1], strict=True)),
        )
    )


def _machine(
    dimensions: int,
    factory: Callable[[], BoxIndexProtocol],
) -> type[RuleBasedStateMachine]:
    bounds_strategy = _box_bounds(dimensions)

    class OptimizedBoxIndexStateMachine(RuleBasedStateMachine):
        def __init__(self) -> None:
            super().__init__()
            self.index = factory()
            self.oracle = BruteBoxIndex(dimensions)
            self.handles: list[tuple[BoxHandle, int]] = []
            self.snapshots: list[tuple[Any, tuple[tuple[int, Any], ...]]] = []

        @rule(bounds=bounds_strategy, data=st.integers(-20, 20))
        def insert(self, bounds, data) -> None:
            lower, upper = bounds
            handle = self.index.insert(Box(lower, upper), data)
            oracle_handle = self.oracle.insert(lower, upper, data)
            self.handles.append((handle, oracle_handle))

        @precondition(lambda self: bool(self.handles))
        @rule(
            position=st.integers(0, 100),
            bounds=bounds_strategy,
            data=st.integers(-20, 20),
        )
        def update(self, position, bounds, data) -> None:
            handle, oracle_handle = self.handles[position % len(self.handles)]
            lower, upper = bounds
            self.index.update(handle, box=Box(lower, upper), data=data)
            self.oracle.update(
                oracle_handle,
                lower=lower,
                upper=upper,
                data=data,
            )

        @precondition(lambda self: bool(self.handles))
        @rule(position=st.integers(0, 100))
        def remove(self, position) -> None:
            location = position % len(self.handles)
            handle, oracle_handle = self.handles.pop(location)
            self.index.remove(handle)
            self.oracle.remove(oracle_handle)
            with pytest.raises(KeyError):
                self.index.get(handle)

        @rule(bounds=bounds_strategy)
        def query(self, bounds) -> None:
            lower, upper = bounds
            observed = [entry.data for entry in self.index.overlaps(Box(lower, upper))]
            expected = [entry.data for entry in self.oracle.overlaps(lower, upper)]
            assert observed == expected

        @rule()
        def snapshot(self) -> None:
            snapshot = self.index.snapshot()
            captured = tuple(
                (entry.handle.sequence, entry.data) for entry in snapshot.entries
            )
            self.snapshots.append((snapshot, captured))

        @invariant()
        def state_matches_oracle(self) -> None:
            observed = self.index.entries()
            expected = tuple(self.oracle.entries.values())
            assert [entry.box.lower for entry in observed] == [
                entry.lower for entry in expected
            ]
            assert [entry.box.upper for entry in observed] == [
                entry.upper for entry in expected
            ]
            assert [entry.data for entry in observed] == [
                entry.data for entry in expected
            ]
            assert self.index.diagnostics().version == self.oracle.version
            for snapshot, captured in self.snapshots:
                current = tuple(
                    (entry.handle.sequence, entry.data) for entry in snapshot.entries
                )
                assert current == captured

    return OptimizedBoxIndexStateMachine


TestBoxIndex2DStateMachine = _machine(2, BoxIndex2D).TestCase
TestBoxIndex3DStateMachine = _machine(3, BoxIndex3D).TestCase
TestBoxIndex4DStateMachine = _machine(4, BoxIndex4D).TestCase
TestBoundedBoxIndexStateMachine = _machine(
    3,
    lambda: BoundedBoxIndex(Box((-5, -5, -5), (6, 6, 6)), (2, 2, 2)),
).TestCase

for test_case in (
    TestBoxIndex2DStateMachine,
    TestBoxIndex3DStateMachine,
    TestBoxIndex4DStateMachine,
    TestBoundedBoxIndexStateMachine,
):
    test_case.settings = settings(
        max_examples=25, stateful_step_count=25, deadline=None
    )
del test_case
