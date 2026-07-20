"""Sparse-grid guardrails and prepared-strategy rollback tests."""

from __future__ import annotations

from typing import Any

import pytest

from treemendous.multidimensional import BoundedBoxIndex, Box, BoxIndex2D


def _bounded(
    *,
    max_total_cells: int = 1_000_000,
    max_cells_per_entry: int = 100_000,
    max_cells_per_query: int = 100_000,
    max_total_postings: int = 1_000_000,
) -> BoundedBoxIndex:
    return BoundedBoxIndex(
        Box((0, 0), (8, 8)),
        (2, 2),
        max_total_cells=max_total_cells,
        max_cells_per_entry=max_cells_per_entry,
        max_cells_per_query=max_cells_per_query,
        max_total_postings=max_total_postings,
    )


def test_bounded_constructor_validates_dimensions_cells_and_all_limits() -> None:
    with pytest.raises(ValueError, match="between 2 and 8"):
        BoundedBoxIndex(Box((0,), (2,)), (1,))
    with pytest.raises(ValueError, match="between 2 and 8"):
        BoundedBoxIndex(Box((0,) * 9, (2,) * 9), (1,) * 9)
    with pytest.raises(TypeError, match="tuple"):
        BoundedBoxIndex(Box((0, 0), (2, 2)), [1, 1])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dimensions"):
        BoundedBoxIndex(Box((0, 0), (2, 2)), (1,))
    for cell_size in ((0, 1), (-1, 1)):
        with pytest.raises(ValueError, match="greater than zero"):
            BoundedBoxIndex(Box((0, 0), (2, 2)), cell_size)
    with pytest.raises(ValueError, match="max_total_cells"):
        BoundedBoxIndex(
            Box((0, 0), (100, 100)),
            (1, 1),
            max_total_cells=9_999,
        )
    invalid_limit_factories = (
        lambda: _bounded(max_total_cells=0),
        lambda: _bounded(max_cells_per_entry=0),
        lambda: _bounded(max_cells_per_query=0),
        lambda: _bounded(max_total_postings=0),
    )
    for make_invalid in invalid_limit_factories:
        with pytest.raises((TypeError, ValueError)):
            make_invalid()


def test_containment_and_per_operation_cell_guards_do_not_mutate() -> None:
    index = _bounded(max_cells_per_entry=3, max_cells_per_query=3)
    before = index.snapshot()

    with pytest.raises(ValueError, match="contained"):
        index.insert(Box((-1, 0), (1, 1)), "outside")
    with pytest.raises(ValueError, match="max_cells_per_entry"):
        index.insert(Box((0, 0), (4, 4)), "too many cells")
    with pytest.raises(ValueError, match="max_cells_per_query"):
        index.overlaps(Box((0, 0), (4, 4)))

    assert index.snapshot() == before
    handle = index.insert(Box((0, 0), (2, 2)), "safe")
    assert handle.sequence == 1


def test_huge_cell_ranges_are_counted_without_combinatorial_allocation() -> None:
    huge = 10**100
    index = BoundedBoxIndex(
        Box((0, 0), (huge, huge)),
        (1, 1),
        max_total_cells=huge * huge,
        max_cells_per_entry=4,
        max_cells_per_query=4,
    )
    enormous = Box((0, 0), (huge, huge))

    with pytest.raises(ValueError, match="max_cells_per_entry"):
        index.insert(enormous)
    with pytest.raises(ValueError, match="max_cells_per_query"):
        index.overlaps(enormous)
    assert len(index) == 0


def test_total_posting_limit_rejects_insert_and_update_before_commit() -> None:
    index = _bounded(
        max_cells_per_entry=16,
        max_total_postings=2,
    )
    first = index.insert(Box((0, 0), (4, 2)), "first")
    before = index.snapshot()
    diagnostics_before = index.diagnostics()

    with pytest.raises(ValueError, match="max_total_postings"):
        index.insert(Box((4, 0), (6, 2)), "rejected")
    with pytest.raises(ValueError, match="max_total_postings"):
        index.update(first, box=Box((0, 0), (4, 4)), data="rejected")

    assert index.snapshot() == before
    assert index.diagnostics() == diagnostics_before
    index.remove(first)
    replacement = index.insert(Box((4, 0), (6, 2)), "replacement")
    assert replacement.sequence == 2


def test_projection_and_grid_diagnostics_track_committed_strategy_state() -> None:
    projection = BoxIndex2D()
    first = projection.insert(Box((0, 0), (2, 2)), "first")
    second = projection.insert(Box((4, 4), (6, 6)), "second")
    projection.update(first, box=Box((1, 1), (3, 3)))
    projection.remove(second)
    projection_diagnostics = projection.diagnostics()
    assert projection_diagnostics.algorithm == "axis_projection"
    expected_projection_sizes = (1, 1)
    assert projection_diagnostics.projection_sizes == expected_projection_sizes
    assert projection_diagnostics.entry_count == 1

    grid = _bounded()
    handle = grid.insert(Box((1, 1), (5, 3)), "value")
    diagnostics = grid.diagnostics()
    assert diagnostics.algorithm == "sparse_grid"
    expected_grid_shape = (4, 4)
    assert diagnostics.grid_shape == expected_grid_shape
    assert diagnostics.posting_count == 6
    assert diagnostics.occupied_cell_count == 6
    assert diagnostics.max_total_cells == 1_000_000
    grid.remove(handle)
    assert grid.diagnostics().posting_count == 0


def test_strategy_prepare_failure_precedes_payload_copy_and_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone_calls = 0

    def cloner(value: Any) -> Any:
        nonlocal clone_calls
        clone_calls += 1
        return value

    index = BoxIndex2D(payload_cloner=cloner)

    def fail_prepare(handle: object, box: object) -> object:
        del handle, box
        raise RuntimeError("prepare failed")

    monkeypatch.setattr(index._strategy, "prepare_insert", fail_prepare)
    with pytest.raises(RuntimeError, match="prepare failed"):
        index.insert(Box((0, 0), (1, 1)), "payload")

    assert clone_calls == 0
    assert len(index) == 0
    assert index.diagnostics().version == 0


def test_query_does_not_change_diagnostics() -> None:
    index = _bounded()
    index.insert(Box((0, 0), (2, 2)), "value")
    before = index.diagnostics()
    for _ in range(10):
        index.overlaps(Box((0, 0), (1, 1)))
    assert index.diagnostics() == before
