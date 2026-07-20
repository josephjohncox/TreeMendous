"""Contracts for canonical interval-record coverage projection."""

from __future__ import annotations

from copy import deepcopy
from random import Random

import pytest

from treemendous.applications._shared.coverage import (
    CoverageSegment,
    coverage_segments,
    coverage_snapshot,
)
from treemendous.applications._shared.interval_records import (
    IntervalRecordIndex,
    RecordHandle,
)


def test_piecewise_coverage_counts_exact_record_id_sets() -> None:
    index = IntervalRecordIndex[str, None](deepcopy)
    first = index.insert("a", 0, 5, None)
    second = index.insert("b", 2, 7, None)
    third = index.insert("c", 5, 6, None)
    coincident = index.insert("d", 2, 7, None)

    assert list(coverage_segments(index.snapshot().records)) == [
        CoverageSegment(0, 2, frozenset({first})),
        CoverageSegment(2, 5, frozenset({first, second, coincident})),
        CoverageSegment(5, 6, frozenset({second, third, coincident})),
        CoverageSegment(6, 7, frozenset({second, coincident})),
    ]
    clipped = coverage_snapshot(index.snapshot().records, start=3, end=6)
    assert [segment.count for segment in clipped.segments] == [3, 3]
    assert clipped.covered_length == 3
    assert clipped.maximum_count == 3


def test_half_open_touching_records_do_not_overlap() -> None:
    index = IntervalRecordIndex[str, None](lambda value: value)
    left = index.insert("a", 0, 2, None)
    right = index.insert("b", 2, 4, None)

    segments = coverage_segments(index.snapshot().records)
    assert list(segments) == [
        CoverageSegment(0, 2, frozenset({left})),
        CoverageSegment(2, 4, frozenset({right})),
    ]


def test_coverage_matches_independent_integer_point_oracle() -> None:
    random = Random(1907)
    index = IntervalRecordIndex[str, int](lambda value: value)
    source: list[tuple[RecordHandle[str], int, int]] = []
    for number in range(80):
        start = random.randrange(-10, 15)
        end = random.randrange(start + 1, 20)
        handle = index.insert(f"owner-{number % 7}", start, end, number)
        source.append((handle, start, end))

    expected = {
        point: frozenset(
            handle for handle, start, end in source if start <= point < end
        )
        for point in range(-10, 20)
    }
    actual: dict[int, frozenset[RecordHandle[str]]] = {
        point: frozenset() for point in range(-10, 20)
    }
    segments = coverage_segments(index.snapshot().records)
    for segment in segments:
        for point in range(segment.start, segment.end):
            actual[point] = segment.record_ids

    assert actual == expected
    assert all(
        left.end <= right.start
        and (left.end < right.start or left.record_ids != right.record_ids)
        for left, right in zip(segments, segments[1:])
    )


def test_empty_clipped_and_invalid_inputs() -> None:
    index = IntervalRecordIndex[str, None](lambda value: value)
    index.insert("owner", 5, 8, None)
    records = index.snapshot().records

    assert not coverage_segments([])
    assert not coverage_segments(records, start=0, end=5)
    with pytest.raises(ValueError, match="supplied together"):
        coverage_segments(records, start=0)
    with pytest.raises(ValueError, match="start < end"):
        coverage_segments(records, start=2, end=2)
    with pytest.raises(ValueError, match="duplicate record identity"):
        coverage_segments((records[0], records[0]))


def test_coverage_segment_rejects_empty_id_set() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CoverageSegment[str](0, 1, frozenset())
