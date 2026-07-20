"""Property and state-machine refinement tests against a finite point oracle."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
)

from tests.oracles.multidimensional.brute_box_index import BruteBoxIndex
from treemendous.multidimensional import Box, BoxHandle, BoxIndex


@st.composite
def box_bounds(
    draw: st.DrawFn, dimensions: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    lower = tuple(
        draw(st.integers(min_value=-2, max_value=3)) for _ in range(dimensions)
    )
    widths = tuple(
        draw(st.integers(min_value=1, max_value=3)) for _ in range(dimensions)
    )
    upper = tuple(
        coordinate + width for coordinate, width in zip(lower, widths, strict=True)
    )
    return lower, upper


@st.composite
def overlap_case(draw: st.DrawFn):
    dimensions = draw(st.integers(min_value=2, max_value=4))
    entries = draw(st.lists(box_bounds(dimensions), min_size=0, max_size=10))
    query = draw(box_bounds(dimensions))
    return dimensions, entries, query


@given(case=overlap_case())
def test_overlap_membership_matches_independent_point_oracle(case) -> None:
    dimensions, entries, query = case
    index = BoxIndex(dimensions)
    oracle = BruteBoxIndex(dimensions)
    for ordinal, (lower, upper) in enumerate(entries):
        index.insert(Box(lower, upper), ordinal)
        oracle.insert(lower, upper, ordinal)
    query_lower, query_upper = query

    observed = [entry.data for entry in index.overlaps(Box(query_lower, query_upper))]
    expected = [entry.data for entry in oracle.overlaps(query_lower, query_upper)]
    assert observed == expected


@given(
    entries=st.lists(box_bounds(4), min_size=0, max_size=8),
    query=box_bounds(4),
    offset=st.tuples(*(st.integers(-5, 5) for _ in range(4))),
)
def test_translation_and_axis_permutation_preserve_overlap_identity(
    entries: list[tuple[tuple[int, ...], tuple[int, ...]]],
    query: tuple[tuple[int, ...], tuple[int, ...]],
    offset: tuple[int, ...],
) -> None:
    original = BoxIndex(4)
    translated = BoxIndex(4)
    permuted = BoxIndex(4)
    for ordinal, (lower, upper) in enumerate(entries):
        original.insert(Box(lower, upper), ordinal)
        translated.insert(
            Box(
                tuple(
                    value + delta for value, delta in zip(lower, offset, strict=True)
                ),
                tuple(
                    value + delta for value, delta in zip(upper, offset, strict=True)
                ),
            ),
            ordinal,
        )
        permuted.insert(Box(tuple(reversed(lower)), tuple(reversed(upper))), ordinal)

    query_lower, query_upper = query
    translated_query = Box(
        tuple(value + delta for value, delta in zip(query_lower, offset, strict=True)),
        tuple(value + delta for value, delta in zip(query_upper, offset, strict=True)),
    )
    original_ids = [entry.data for entry in original.overlaps(Box(*query))]
    translated_ids = [entry.data for entry in translated.overlaps(translated_query)]
    permuted_ids = [
        entry.data
        for entry in permuted.overlaps(
            Box(tuple(reversed(query_lower)), tuple(reversed(query_upper)))
        )
    ]
    assert original_ids == translated_ids == permuted_ids


class BoxIndexStateMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.index = BoxIndex(2)
        self.oracle = BruteBoxIndex(2)
        self.handles: list[tuple[BoxHandle, int]] = []
        self.stale: list[BoxHandle] = []
        self.snapshots: list[tuple[Any, tuple[tuple[int, Any], ...]]] = []

    @rule(
        bounds=box_bounds(2),
        data=st.one_of(st.none(), st.integers(-10, 10)),
    )
    def insert(self, bounds, data) -> None:
        lower, upper = bounds
        handle = self.index.insert(Box(lower, upper), data)
        oracle_handle = self.oracle.insert(lower, upper, data)
        self.handles.append((handle, oracle_handle))

    @precondition(lambda self: bool(self.handles))
    @rule(
        position=st.integers(min_value=0, max_value=100),
        bounds=box_bounds(2),
        data=st.one_of(st.none(), st.integers(-10, 10)),
    )
    def update(self, position, bounds, data) -> None:
        handle, oracle_handle = self.handles[position % len(self.handles)]
        lower, upper = bounds
        observed = self.index.update(handle, box=Box(lower, upper), data=data)
        self.oracle.update(
            oracle_handle,
            lower=lower,
            upper=upper,
            data=data,
        )
        assert observed.handle == handle

    @precondition(lambda self: bool(self.handles))
    @rule(position=st.integers(min_value=0, max_value=100))
    def remove(self, position) -> None:
        location = position % len(self.handles)
        handle, oracle_handle = self.handles.pop(location)
        observed = self.index.remove(handle)
        expected = self.oracle.remove(oracle_handle)
        assert observed.data == expected.data
        self.stale.append(handle)

    @rule(bounds=box_bounds(2))
    def query(self, bounds) -> None:
        lower, upper = bounds
        observed = [entry.data for entry in self.index.overlaps(Box(lower, upper))]
        expected = [entry.data for entry in self.oracle.overlaps(lower, upper)]
        assert observed == expected

    @rule()
    def capture_snapshot(self) -> None:
        snapshot = self.index.snapshot()
        expected = tuple(
            (entry.handle.sequence, entry.data) for entry in snapshot.entries
        )
        self.snapshots.append((snapshot, expected))

    @precondition(lambda self: bool(self.stale))
    @rule(position=st.integers(min_value=0, max_value=100))
    def stale_handle_fails(self, position) -> None:
        handle = self.stale[position % len(self.stale)]
        with pytest.raises(KeyError):
            self.index.get(handle)

    @invariant()
    def implementation_refines_oracle(self) -> None:
        observed = self.index.entries()
        expected = tuple(self.oracle.entries.values())
        assert len(observed) == len(expected)
        assert [entry.handle for entry in observed] == [
            pair[0] for pair in self.handles
        ]
        assert [entry.box.lower for entry in observed] == [
            entry.lower for entry in expected
        ]
        assert [entry.box.upper for entry in observed] == [
            entry.upper for entry in expected
        ]
        assert [entry.data for entry in observed] == [entry.data for entry in expected]
        diagnostics = self.index.diagnostics()
        assert diagnostics.version == self.oracle.version
        assert diagnostics.entry_count == len(expected)
        for snapshot, captured in self.snapshots:
            current = tuple(
                (entry.handle.sequence, entry.data) for entry in snapshot.entries
            )
            assert current == captured


TestBoxIndexStateMachine = BoxIndexStateMachine.TestCase
