"""Correctness-checked smoke workload for genetic search."""

from tests.oracles.applications.partitioning.genetic_search import expected_ranking
from treemendous.applications.partitioning.genetic_search import GeneticSearchEngine


def _fitness(value: str) -> float:
    return value.count("1") * 1.0


def run_smoke() -> int:
    population = tuple(f"{value:08b}" for value in range(32))
    engine = GeneticSearchEngine(population, _fitness, generations=8, seed=19)
    first = engine.step()
    if first.ranking != expected_ranking(population, _fitness):
        raise AssertionError("genetic initial ranking differs from oracle")
    history = engine.run()
    if len(history) != 8:
        raise AssertionError("genetic smoke did not checkpoint every generation")
    return len(history)
