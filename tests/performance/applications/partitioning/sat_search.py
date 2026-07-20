"""Correctness-checked smoke workload for CNF SAT search."""

from tests.oracles.applications.partitioning.sat_search import expected_ordinals
from treemendous.applications.partitioning.sat_search import SatSearchEngine


def run_smoke() -> int:
    clauses = ((1, 2), (-1, 3), (4, -2), (5, -3), (6,))
    engine = SatSearchEngine(6, clauses, prefix_bits=3)
    observed = tuple(item.ordinal for item in engine.run(shard_size=2))
    if observed != expected_ordinals(6, clauses):
        raise AssertionError("SAT smoke differs from truth-table oracle")
    return len(observed)
