from treemendous.basic.boundary import IntervalManager
from treemendous.basic.boundary_summary import BoundarySummaryManager


def test_boundary_can_merge_adjacent_with_payloads():
    def can_merge(left, right):
        return left == right

    manager = IntervalManager(can_merge=can_merge)
    manager.release_interval(0, 10, data="A")
    manager.release_interval(10, 20, data="B")
    manager.release_interval(20, 30, data="B")

    intervals = manager.get_intervals()
    assert len(intervals) == 2

    first, second = intervals
    assert (first.start, first.end, first.data) == (0, 10, "A")
    assert (second.start, second.end, second.data) == (10, 30, "B")


def test_boundary_split_fn_on_reserve():
    def split_fn(data, old_start, old_end, new_start, new_end):
        return f"{data}:{new_start}-{new_end}"

    manager = BoundarySummaryManager(split_fn=split_fn)
    manager.release_interval(0, 100, data="pool")
    manager.reserve_interval(25, 75)

    intervals = manager.get_intervals()
    assert len(intervals) == 2

    first, second = intervals
    assert (first.start, first.end, first.data) == (0, 25, "pool:0-25")
    assert (second.start, second.end, second.data) == (75, 100, "pool:75-100")
