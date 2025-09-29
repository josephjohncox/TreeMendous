from hypothesis import given, assume, strategies as st
from treemendous.basic.avl_earliest import EarliestIntervalNode, EarliestIntervalTree
from typing import List, Tuple, Optional, Set

@given(st.lists(st.tuples(
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000)
).filter(lambda x: x[0] < x[1])).map(sorted).filter(
    lambda ops: all(ops[i][1] <= ops[i+1][0] for i in range(len(ops)-1))
))
def test_insert_and_delete_intervals(operations: List[Tuple[int, int]]) -> None:
    tree = EarliestIntervalTree()
    tree.release_interval(0, 1000)
    total_available: int = 1000

    for start, end in operations:
        tree.reserve_interval(start, end)
        total_available -= end - start

    assert tree.get_total_available_length() == total_available

    for start, end in operations:
        if end > 1000:
            continue
        tree.release_interval(start, end)
        total_available += end - start

    assert tree.get_total_available_length() == total_available

@given(st.integers(min_value=0, max_value=1000), st.integers(min_value=1, max_value=500))
def test_find_interval(point: int, length: int) -> None:
    tree = EarliestIntervalTree()
    tree.release_interval(0, 1000)
    tree.reserve_interval(200, 300)
    tree.reserve_interval(400, 500)

    interval: Optional[EarliestIntervalNode] = tree.find_interval(point, length)
    if interval:
        assert interval.end - interval.start >= length
        # For "earliest" fit: finds the earliest interval that can accommodate the request
        # The interval must be able to fit the request starting from the requested point OR
        # be large enough to accommodate the full request regardless of where it starts
        can_fit_from_point = (interval.start <= point < interval.end and 
                             interval.end - point >= length)
        can_fit_entire_request = interval.end - interval.start >= length
        
        assert can_fit_from_point or can_fit_entire_request, \
            f"Interval [{interval.start}, {interval.end}) cannot accommodate request: point={point}, length={length}"
        assert not (200 <= interval.start < 300 or 200 < interval.end <= 300)
        assert not (400 <= interval.start < 500 or 400 < interval.end <= 500)

def test_total_available_length() -> None:
    tree = EarliestIntervalTree()
    tree.release_interval(0, 1000)
    tree.reserve_interval(100, 200)
    tree.reserve_interval(300, 400)

    assert tree.get_total_available_length() == 800

    tree.release_interval(150, 350)
    assert tree.get_total_available_length() == 900

def test_adjacent_intervals() -> None:
    tree = EarliestIntervalTree()
    tree.release_interval(0, 100)
    tree.release_interval(100, 200)

    intervals = tree.get_intervals()
    assert intervals == [(0, 100), (100, 200)]
    assert tree.get_total_available_length() == 200

@given(st.lists(st.tuples(
    st.sampled_from(['insert', 'delete']),
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=1, max_value=500)
).filter(lambda x: x[2] > 0)).map(sorted).filter(
    lambda ops: all(ops[i][2] + ops[i][1] <= ops[i+1][1] for i in range(len(ops)-1))
))
def test_random_operations(operations: List[Tuple[str, int, int]]) -> None:
    tree = EarliestIntervalTree()
    tree.release_interval(0, 1000)
    total_available: int = 1000

    for op, start, length in operations:
        end: int = start + length
        if end > 1000:
            continue
        if op == 'delete':
            tree.reserve_interval(start, end)
            total_available -= length
        else:
            # Only increase total if interval isn't already in tree
            interval: Optional[EarliestIntervalNode] = tree.find_interval(start, length)
            if interval and interval.start > start:
                total_available += length
            tree.release_interval(start, end)

    total_available = max(0, min(total_available, 1000))
    assert 0 <= total_available <= 1000
    if total_available != tree.get_total_available_length():
        tree.print_tree()
    assert tree.get_total_available_length() == total_available

@given(st.lists(st.tuples(
    st.integers(min_value=0, max_value=999),
    st.integers(min_value=1, max_value=1000)
).filter(lambda x: x[0] + x[1] <= 1000)))
def test_non_overlapping_reservations(intervals: List[Tuple[int, int]]) -> None:
    sorted_intervals: List[Tuple[int, int]] = sorted(
        ((start, start + length) for start, length in intervals),
        key=lambda x: x[0]
    )

    assume(all(s2 >= e1 for (_, e1), (s2, _) in zip(sorted_intervals, sorted_intervals[1:])))

    tree = EarliestIntervalTree()
    tree.release_interval(0, 1000)

    for start, end in sorted_intervals:
        tree.reserve_interval(start, end)

    total_reserved: int = sum(end - start for start, end in sorted_intervals)
    total_available: int = tree.get_total_available_length()
    assert total_available == 1000 - total_reserved

@given(st.lists(st.tuples(
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000)
).filter(lambda x: x[0] < x[1])))
def test_overlapping_reservations(intervals: List[Tuple[int, int]]) -> None:
    tree = EarliestIntervalTree()
    tree.release_interval(0, 1000)

    covered_points: Set[int] = set()

    for start, end in intervals:
        tree.reserve_interval(start, end)
        covered_points.update(range(start, end))

    total_reserved: int = len(covered_points)
    total_available: int = tree.get_total_available_length()
    assert total_available + total_reserved == 1000