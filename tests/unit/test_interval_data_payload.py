from typing import Any, List, Tuple

import pytest

from treemendous.basic.boundary import IntervalManager
from treemendous.basic.boundary_summary import BoundarySummaryManager
from treemendous.basic.summary import SummaryIntervalTree
from treemendous.basic.treap import IntervalTreap
from treemendous.basic.avl_earliest import EarliestIntervalTree


def normalize_intervals(intervals: List[Any]) -> List[Tuple[int, int, Any]]:
    """Normalize intervals to (start, end, data) tuples."""
    normalized: List[Tuple[int, int, Any]] = []
    for interval in intervals:
        if hasattr(interval, "start") and hasattr(interval, "end"):
            normalized.append((interval.start, interval.end, getattr(interval, "data", None)))
        elif isinstance(interval, tuple):
            if len(interval) == 3:
                start, end, data = interval
                normalized.append((start, end, data))
            elif len(interval) == 2:
                start, end = interval
                normalized.append((start, end, None))
    return normalized


@pytest.mark.parametrize(
    "factory",
    [
        IntervalManager,
        BoundarySummaryManager,
        SummaryIntervalTree,
        lambda: IntervalTreap(random_seed=42),
        EarliestIntervalTree,
    ],
)
def test_interval_data_roundtrip(factory) -> None:
    tree = factory()
    payload_a = {"job_id": "A", "priority": 1}
    payload_b = {"job_id": "B", "priority": 2}
    
    tree.release_interval(0, 10, payload_a)
    tree.release_interval(20, 30, payload_b)
    
    intervals = normalize_intervals(tree.get_intervals())
    data_values = {data["job_id"] for _, _, data in intervals if isinstance(data, dict)}
    
    assert {"A", "B"}.issubset(data_values)
