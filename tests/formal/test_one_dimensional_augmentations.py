"""Executable checks for fit-pruning certificates and their failure modes."""

from __future__ import annotations

from typing import cast

from tests.formal.one_dimensional_model import (
    all_point_sets,
    runs,
    safely_reject_fit,
)
from treemendous.basic.avl_earliest import EarliestIntervalNode, EarliestIntervalTree
from treemendous.basic.summary import (
    SummaryIntervalNode,
    SummaryIntervalTree,
    TreeSummary,
)


def _fits(spans: tuple[tuple[int, int], ...], point: int, length: int) -> bool:
    return any(max(point, start) + length <= end for start, end in spans)


def test_exact_and_overestimated_bounds_never_prune_a_real_fit() -> None:
    for free in all_point_sets(6):
        spans = runs(free)
        true_length = max((end - start for start, end in spans), default=0)
        true_end = max((end for _, end in spans), default=0)
        for point in range(7):
            for length in range(1, 7):
                if _fits(spans, point, length):
                    assert not safely_reject_fit(
                        request_start=point,
                        length=length,
                        upper_length=true_length,
                        upper_end=true_end,
                    )
                    assert not safely_reject_fit(
                        request_start=point,
                        length=length,
                        upper_length=true_length + 3,
                        upper_end=true_end + 3,
                    )


def test_underestimated_length_or_end_can_create_false_negatives() -> None:
    spans = ((5, 9),)
    assert _fits(spans, 8, 1)
    assert safely_reject_fit(
        request_start=8,
        length=1,
        upper_length=4,
        upper_end=8,
    )
    assert safely_reject_fit(
        request_start=5,
        length=4,
        upper_length=3,
        upper_end=9,
    )


def _assert_earliest_augmentations(
    node: EarliestIntervalNode | None,
) -> tuple[int, int, int | None, int | None, int]:
    if node is None:
        return 0, 0, None, None, 0
    left_height, left_total, left_min, left_max, left_length = (
        _assert_earliest_augmentations(cast(EarliestIntervalNode | None, node.left))
    )
    right_height, right_total, right_min, right_max, right_length = (
        _assert_earliest_augmentations(cast(EarliestIntervalNode | None, node.right))
    )
    assert abs(left_height - right_height) <= 1
    assert node.height == 1 + max(left_height, right_height)
    assert node.total_length == node.length + left_total + right_total
    expected_min = min(
        value for value in (left_min, node.start, right_min) if value is not None
    )
    expected_max = max(
        value for value in (left_max, node.end, right_max) if value is not None
    )
    expected_length = max(left_length, node.length, right_length)
    assert node.min_start == expected_min
    assert node.max_end == expected_max
    assert node.max_length == expected_length
    return (
        node.height,
        node.total_length,
        expected_min,
        expected_max,
        expected_length,
    )


def _assert_summary_augmentations(node: SummaryIntervalNode | None) -> tuple[int, int]:
    if node is None:
        return 0, 0
    left_height, left_total = _assert_summary_augmentations(node.left)
    right_height, right_total = _assert_summary_augmentations(node.right)
    assert abs(left_height - right_height) <= 1
    assert node.height == 1 + max(left_height, right_height)
    assert node.total_length == node.length + left_total + right_total
    expected = TreeSummary.merge(
        node.left.summary if node.left else None,
        node.right.summary if node.right else None,
        TreeSummary.from_interval(node.start, node.end),
    )
    assert node.summary == expected
    return node.height, node.total_length


def test_production_avl_augmentations_survive_bulk_range_deletion() -> None:
    earliest = EarliestIntervalTree()
    summary = SummaryIntervalTree()
    for index in range(16):
        earliest.release_interval(2 * index, 2 * index + 1)
        summary.release_interval(2 * index, 2 * index + 1)

    earliest.reserve_interval(16, 33)
    summary.reserve_interval(16, 33)

    _assert_earliest_augmentations(earliest.root)
    _assert_summary_augmentations(summary.root)
    assert earliest.find_interval(0, 1) is not None
    summary_fit = summary.find_interval(0, 1)
    assert summary_fit is not None
    assert summary_fit[0] == 0
    assert summary_fit[1] == 1


def test_pruning_bounds_are_rejection_not_success_certificates() -> None:
    spans = ((0, 2), (4, 6))
    point = 1
    length = 3
    true_length = 2
    true_end = 6
    assert not _fits(spans, point, length)
    assert safely_reject_fit(
        request_start=point,
        length=length,
        upper_length=true_length,
        upper_end=true_end,
    )

    # Passing both bounds still does not prove success: their maxima can come
    # from different candidates.
    spans = ((0, 10), (100, 101))
    assert not _fits(spans, 50, 5)
    assert not safely_reject_fit(
        request_start=50,
        length=5,
        upper_length=10,
        upper_end=101,
    )
