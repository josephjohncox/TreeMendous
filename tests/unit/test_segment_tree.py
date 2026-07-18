from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from treemendous.basic.segment import SegmentTree, SegmentTreeNode
from treemendous.domain import Span

VALID_SPAN = st.tuples(
    st.integers(min_value=0, max_value=15),
    st.integers(min_value=1, max_value=16),
).filter(lambda bounds: bounds[0] < bounds[1])


def _free_runs(free: list[bool]) -> list[Span]:
    runs: list[Span] = []
    start: int | None = None
    for point, available in enumerate([*free, False]):
        if available and start is None:
            start = point
        elif not available and start is not None:
            runs.append(Span(start, point))
            start = None
    return runs


def _spans(tree: SegmentTree) -> list[Span]:
    return [item.span for item in tree.get_intervals()]


def test_segment_tree_is_lazy_and_build_is_a_compatibility_noop() -> None:
    tree = SegmentTree(0, 16)
    assert _spans(tree) == [Span(0, 16)]
    assert tree.get_total_available_length() == 16
    assert tree.root is not None and tree.root.left is None

    tree.build()

    assert _spans(tree) == [Span(0, 16)]
    assert tree.root is not None and tree.root.left is None


def test_segment_tree_updates_are_canonical_and_collapse_full_ranges() -> None:
    tree = SegmentTree(0, 16)
    tree.schedule_interval(3, 7)
    assert _spans(tree) == [Span(0, 3), Span(7, 16)]
    assert tree.get_total_available_length() == 12

    tree.schedule_interval(6, 10)
    assert _spans(tree) == [Span(0, 3), Span(10, 16)]
    assert tree.get_total_available_length() == 9

    tree.unschedule_interval(5, 8)
    assert _spans(tree) == [Span(0, 3), Span(5, 8), Span(10, 16)]
    assert tree.get_total_available_length() == 12

    tree.unschedule_interval(0, 16)
    assert _spans(tree) == [Span(0, 16)]
    assert tree.root is not None and tree.root.left is None

    tree.schedule_interval(0, 16)
    assert not tree.get_intervals()
    assert tree.get_total_available_length() == 0


def test_segment_tree_rejects_invalid_spans_atomically() -> None:
    tree = SegmentTree(0, 8)
    tree.schedule_interval(2, 4)
    before = (_spans(tree), tree.get_total_available_length())

    for operation in (tree.schedule_interval, tree.unschedule_interval):
        for bounds in ((3, 3), (5, 4)):
            with pytest.raises(ValueError):
                operation(*bounds)
            assert _spans(tree) == before[0]
            assert tree.get_total_available_length() == before[1]


def test_segment_tree_clips_valid_updates_to_its_managed_extent() -> None:
    tree = SegmentTree(0, 8)
    tree.schedule_interval(-5, -1)
    assert _spans(tree) == [Span(0, 8)]

    tree.schedule_interval(-2, 2)
    assert _spans(tree) == [Span(2, 8)]
    tree.unschedule_interval(7, 12)
    assert _spans(tree) == [Span(2, 8)]


def test_segment_node_aggregates_and_tree_printing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parent = SegmentTreeNode(0, 2)
    parent.is_full = False
    parent.update_node()
    assert parent.total_length == 0

    parent.left = SegmentTreeNode(0, 1)
    parent.right = SegmentTreeNode(1, 2)
    parent.left.is_full = False
    parent.left.total_length = 0
    parent.update_node()
    assert parent.total_length == 1
    assert not parent.is_full

    tree = SegmentTree(0, 4)
    tree.schedule_interval(1, 2)
    tree.print_tree()
    output = capsys.readouterr().out
    assert "0-4" in output
    assert "is_full=False" in output


@given(st.lists(st.tuples(VALID_SPAN, st.booleans()), max_size=40))
def test_segment_tree_matches_small_domain_oracle(
    operations: list[tuple[tuple[int, int], bool]],
) -> None:
    tree = SegmentTree(0, 16)
    free = [True] * 16

    for (start, end), make_available in operations:
        if make_available:
            tree.unschedule_interval(start, end)
        else:
            tree.schedule_interval(start, end)
        free[start:end] = [make_available] * (end - start)
        assert _spans(tree) == _free_runs(free)
        assert tree.get_total_available_length() == sum(free)
