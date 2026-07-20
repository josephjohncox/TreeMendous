"""Search-index merge engine contracts."""

import pytest

from tests.oracles.applications.partitioning.index_merge import expected_index
from treemendous.applications.partitioning.index_merge import IndexMergeEngine


def test_term_bands_merge_sorted_postings_with_deduplication() -> None:
    segments = ({"a": (1, 2, 2), "c": (9,)}, {"a": (2, 3), "b": (4,)})
    engine = IndexMergeEngine(segments)
    observed = tuple((item.term, item.postings) for item in engine.run(band_size=1))
    expected_terms = ("a", "b", "c")
    assert observed == expected_index(segments)
    assert engine.snapshot().terms == expected_terms


def test_index_merge_rejects_unsorted_and_invalid_postings() -> None:
    with pytest.raises(ValueError, match="sorted"):
        IndexMergeEngine(({"a": (2, 1)},))
    with pytest.raises(ValueError, match="non-negative"):
        IndexMergeEngine(({"a": (-1,)},))
