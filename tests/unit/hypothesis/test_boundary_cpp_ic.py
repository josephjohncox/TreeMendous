from hypothesis import assume, given, strategies as st
from treemendous.cpp.boundary import ICIntervalManager as IntervalManager
from typing import List, Tuple, Optional

@given(st.lists(st.tuples(
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000)
).filter(lambda x: x[0] < x[1])).map(sorted).filter(
    lambda ops: all(ops[i][1] <= ops[i+1][0] for i in range(len(ops)-1))
))
def test_reserve_and_release_intervals(operations: List[Tuple[int, int]]) -> None:
    manager = IntervalManager()
    manager.release_interval(0, 1000)

    total_reserved = 0

    for start, end in operations:
        manager.reserve_interval(start, end)
        total_reserved += end - start

    total_available = manager.get_total_available_length()
    assert total_available == 1000 - total_reserved

    for start, end in operations:
        manager.release_interval(start, end)

    assert manager.get_total_available_length() == 1000

@given(st.integers(min_value=0, max_value=1000), st.integers(min_value=1, max_value=500))
def test_find_interval(point: int, length: int) -> None:
    manager = IntervalManager()
    manager.release_interval(0, 1000)
    manager.reserve_interval(200, 300)
    manager.reserve_interval(400, 500)

    interval: Optional[Tuple[int, int]] = manager.find_interval(point, length)
    if interval:
        assert interval[1] - interval[0] >= length
        assert interval[0] >= point
        assert not (200 <= interval[0] < 300 or 200 < interval[1] <= 300)
        assert not (400 <= interval[0] < 500 or 400 < interval[1] <= 500)

def test_total_available_length() -> None:
    manager = IntervalManager()
    manager.release_interval(0, 1000)
    manager.reserve_interval(100, 200)
    manager.reserve_interval(300, 400)

    assert manager.get_total_available_length() == 800

    manager.release_interval(150, 350)
    assert manager.get_total_available_length() == 900

def test_interval_merging() -> None:
    manager = IntervalManager()
    manager.release_interval(0, 100)
    manager.reserve_interval(10, 20)
    manager.reserve_interval(30, 40)
    manager.release_interval(15, 35)

    expected_intervals = [(0, 10), (15, 35), (40, 100)]
    intervals: List[Tuple[int, int]] = []
    for start, end in manager.get_intervals():
        intervals.append((start, end))

    assert intervals == expected_intervals

def test_empty_manager() -> None:
    manager = IntervalManager()
    assert manager.get_total_available_length() == 0
    assert manager.find_interval(0, 10) is None

def test_adjacent_intervals() -> None:
    manager = IntervalManager()
    manager.release_interval(0, 100)
    manager.release_interval(100, 200)
    
    # Should merge into single interval
    intervals = [(s,e) for s,e in manager.get_intervals()]
    assert intervals == [(0, 200)]
    assert manager.get_total_available_length() == 200

def test_overlapping_release() -> None:
    manager = IntervalManager()
    manager.release_interval(0, 100)
    manager.reserve_interval(25, 75)
    
    # Release overlapping both sides
    manager.release_interval(20, 80)
    
    intervals = [(s,e) for s,e in manager.get_intervals()]
    assert intervals == [(0, 100)]
    assert manager.get_total_available_length() == 100

@given(st.integers(min_value=0, max_value=100), st.integers(min_value=0, max_value=100))
def test_reserve_empty_interval(start: int, end: int) -> None:
    manager = IntervalManager()
    manager.release_interval(0, 100)
    manager.reserve_interval(start, start)  # Empty interval
    manager.reserve_interval(end, end)  # Empty interval
    
    # Should not affect available length
    assert manager.get_total_available_length() == 100

@given(st.lists(st.tuples(
    st.sampled_from(['reserve', 'release']),
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000)
).filter(lambda x: x[1] < x[2])).map(
    lambda ops: [(op, start, end) for op, start, end in ops]
))
def test_random_operations(operations: List[Tuple[str, int, int]]) -> None:
    manager: IntervalManager = IntervalManager()
    manager.release_interval(0, 1000)

    for op, start, end in operations:
        if op == 'reserve':
            manager.reserve_interval(start, end)
        else:
            manager.release_interval(start, end)

    total_available: int = manager.get_total_available_length()
    assert 0 <= total_available <= 1000


@given(st.integers(min_value=0, max_value=999), st.integers(min_value=1, max_value=1000))
def test_full_reserve_and_release(start: int, length: int) -> None:
    end: int = start + length
    assume(end <= 1000)

    manager: IntervalManager = IntervalManager()
    manager.release_interval(0, 1000)
    manager.reserve_interval(start, end)
    assert 1000 - (end - start) == manager.get_total_available_length()

    manager.release_interval(start, end)
    assert manager.get_total_available_length() == 1000

@given(st.lists(st.integers(min_value=0, max_value=999)))
def test_single_point_reservations(points: List[int]) -> None:
    manager: IntervalManager = IntervalManager()
    manager.release_interval(0, 1000)

    for point in points:
        manager.reserve_interval(point, point + 1)

    total_reserved: int = len(set(points))
    total_available: int = manager.get_total_available_length()
    assert total_available == 1000 - total_reserved

@given(st.lists(st.tuples(
    st.integers(min_value=0, max_value=999),
    st.integers(min_value=1, max_value=1000)
).filter(lambda x: x[0] + x[1] <= 1000)))
def test_non_overlapping_reservations(intervals: List[Tuple[int, int]]) -> None:
    sorted_intervals: List[Tuple[int, int]] = sorted(
        ((start, start + length) for start, length in intervals),
        key=lambda x: x[0]
    )

    # Ensure intervals do not overlap
    assume(all(s2 >= e1 for (_, e1), (s2, _) in zip(sorted_intervals, sorted_intervals[1:])))

    manager: IntervalManager = IntervalManager()
    manager.release_interval(0, 1000)

    for start, end in sorted_intervals:
        manager.reserve_interval(start, end)

    total_reserved: int = sum(end - start for start, end in sorted_intervals)
    total_available: int = manager.get_total_available_length()
    assert total_available == 1000 - total_reserved

@given(st.lists(st.tuples(
    st.integers(min_value=0, max_value=1000),
    st.integers(min_value=0, max_value=1000)
).filter(lambda x: x[0] < x[1])))
def test_overlapping_reservations(intervals: List[Tuple[int, int]]) -> None:
    manager: IntervalManager = IntervalManager()
    manager.release_interval(0, 1000)

    # Calculate total length and subtract intersections
    # Calculate total reserved length by tracking covered points
    covered_points: set[int] = set()
    sorted_intervals = sorted(intervals)
    
    # Do the reservations and track covered points
    for start, end in sorted_intervals:
        manager.reserve_interval(start, end)
        covered_points.update(range(start, end))
            
    # Total reserved is number of unique points covered
    total_reserved: int = len(covered_points)
    total_available: int = manager.get_total_available_length()
    assert total_available + total_reserved == 1000