"""SAT-search engine contracts."""

import pytest

from tests.oracles.applications.partitioning.sat_search import expected_ordinals
from treemendous.applications.partitioning.sat_search import SatSearchEngine


def test_prefix_partitioning_matches_exhaustive_truth_table() -> None:
    clauses = ((1, -2), (2, 3))
    engine = SatSearchEngine(3, clauses, prefix_bits=2)
    observed = tuple(item.ordinal for item in engine.run(shard_size=1))
    assert observed == expected_ordinals(3, clauses)
    assert engine.snapshot().solutions == engine.run()


def test_sat_search_validates_literals_and_clauses() -> None:
    with pytest.raises(ValueError, match="empty"):
        SatSearchEngine(2, ((),))
    with pytest.raises(ValueError, match="signed"):
        SatSearchEngine(2, ((3,),))
