"""Exhaustive small-universe refinement checks for BoxIndex."""

from __future__ import annotations

from tests.formal.finite_model import finite_boxes
from tests.oracles.multidimensional.brute_box_index import BruteBoxIndex
from treemendous.multidimensional import Box, BoxIndex


def test_all_two_dimensional_overlap_queries_refine_point_set_oracle() -> None:
    bounds = finite_boxes(dimensions=2, extent=2)
    index = BoxIndex(2)
    oracle = BruteBoxIndex(2)
    for ordinal, (lower, upper) in enumerate(bounds):
        index.insert(Box(lower, upper), ordinal)
        oracle.insert(lower, upper, ordinal)
        # Duplicate identity is part of the denotation, not an optimization detail.
        index.insert(Box(lower, upper), -ordinal - 1)
        oracle.insert(lower, upper, -ordinal - 1)

    for lower, upper in bounds:
        observed = [entry.data for entry in index.overlaps(Box(lower, upper))]
        expected = [entry.data for entry in oracle.overlaps(lower, upper)]
        assert observed == expected


def test_bounded_update_remove_trace_preserves_identity_and_version() -> None:
    bounds = finite_boxes(dimensions=2, extent=2)
    for first_bounds in bounds:
        for replacement_bounds in bounds:
            index = BoxIndex(2)
            oracle = BruteBoxIndex(2)
            handle = index.insert(Box(*first_bounds), "value")
            oracle_handle = oracle.insert(*first_bounds, "value")

            index.update(handle, box=Box(*replacement_bounds), data="updated")
            oracle.update(
                oracle_handle,
                lower=replacement_bounds[0],
                upper=replacement_bounds[1],
                data="updated",
            )
            observed = index.get(handle)
            expected = oracle.entries[oracle_handle]
            assert observed.box.lower == expected.lower
            assert observed.box.upper == expected.upper
            assert observed.data == expected.data
            assert index.diagnostics().version == oracle.version

            index.remove(handle)
            oracle.remove(oracle_handle)
            assert len(index) == 0
            assert index.diagnostics().version == oracle.version
