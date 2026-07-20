from __future__ import annotations

import pytest


def test_optimized_boundary_summary_optional_results_are_python_values() -> None:
    module = pytest.importorskip("treemendous.cpp.boundary_summary_optimized")
    manager = module.BoundarySummaryManager()
    manager.release_interval(0, 100)
    manager.reserve_interval(20, 40)

    first_fit = manager.find_interval(0, 10)
    best_fit = manager.find_best_fit(10)
    largest = manager.find_largest_available()
    assert first_fit is not None and list(first_fit) == [0, 10]
    assert best_fit is not None and list(best_fit) == [0, 10]
    assert largest is not None and list(largest) == [40, 100]
    assert manager.find_interval(100, 1) is None

    fragmented = module.BoundarySummaryManager()
    for start, end in ((0, 5), (10, 12), (20, 30)):
        fragmented.release_interval(start, end)
    later_fit = fragmented.find_interval(0, 6)
    assert later_fit is not None and list(later_fit) == [20, 26]

    empty = module.BoundarySummaryManager()
    assert empty.find_best_fit(1) is None
    assert empty.find_largest_available() is None


def test_experimental_treap_empty_optional_results_are_none() -> None:
    module = pytest.importorskip("treemendous.cpp.treap")
    manager = module.IntervalTreap(42)

    assert manager.find_interval(0, 1) is None
    assert manager.sample_random_interval() is None
    manager.release_interval(10, 20)
    first_fit = manager.find_interval(0, 2)
    sample = manager.sample_random_interval()
    assert first_fit is not None and list(first_fit) == [10, 12]
    assert sample is not None and list(sample) == [10, 20]

    fragmented = module.IntervalTreap(42)
    for start, end in ((0, 5), (10, 12), (20, 30)):
        fragmented.release_interval(start, end)
    later_fit = fragmented.find_interval(0, 6)
    assert later_fit is not None and list(later_fit) == [20, 26]
