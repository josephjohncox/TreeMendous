"""Correctness-checked smoke workload for search-index merge."""

from tests.oracles.applications.partitioning.index_merge import expected_index
from treemendous.applications.partitioning.index_merge import IndexMergeEngine


def run_smoke() -> int:
    segments = tuple({f"term-{term:03}": tuple(range(segment, 100, 5)) for term in range(segment, 100, 3)} for segment in range(5))
    engine = IndexMergeEngine(segments)
    observed = tuple((item.term, item.postings) for item in engine.run(band_size=7))
    if observed != expected_index(segments):
        raise AssertionError("index merge smoke differs from set/sort oracle")
    return len(observed)
