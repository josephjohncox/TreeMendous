"""Genetic-search engine contracts."""

from dataclasses import replace
from math import nan

import pytest

from tests.oracles.applications.partitioning.genetic_search import expected_ranking
from treemendous.applications._shared.claiming import ClaimInvariantError
from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.partitioning.genetic_search import (
    GeneticGeneration,
    GeneticSearchEngine,
)


def _fitness(value: str) -> float:
    return value.count("1") * 1.0


def _zero_fitness(_: str) -> float:
    return 0.0


def _invalid_fitness(_: str) -> float:
    return nan


def test_seeded_generations_are_reproducible_and_checkpointable() -> None:
    population = ("000", "001", "110", "111")
    fitness = _fitness
    clock = LogicalClock()
    left = GeneticSearchEngine(
        population,
        fitness,
        generations=3,
        seed=7,
        mutation_rate=0.2,
        clock=clock,
    )
    right = GeneticSearchEngine(
        population, fitness, generations=3, seed=7, mutation_rate=0.2
    )
    assert left.step().ranking == expected_ranking(population, fitness)
    restored = GeneticSearchEngine.from_checkpoint(
        left.checkpoint(), fitness=fitness, clock=clock
    )
    left.run()
    restored.run()
    right.run()
    assert left.snapshot().population == restored.snapshot().population
    assert left.snapshot().population == right.snapshot().population
    assert left.checkpoint().random_state == right.checkpoint().random_state


def test_restore_rejects_contradictory_application_and_runtime_progress() -> None:
    population = ("00", "11")
    engine = GeneticSearchEngine(population, _fitness, generations=2)
    checkpoint = engine.checkpoint()
    record = GeneticGeneration(0, population, expected_ranking(population, _fitness))
    mixed = replace(checkpoint, generation=1, history=(record,))

    with pytest.raises(ClaimInvariantError, match="completion count"):
        GeneticSearchEngine.from_checkpoint(
            mixed, fitness=_fitness, clock=LogicalClock()
        )
    with pytest.raises(ValueError, match="mutation_rate"):
        GeneticSearchEngine.from_checkpoint(
            replace(checkpoint, mutation_rate=2.0),
            fitness=_fitness,
            clock=LogicalClock(),
        )
    with pytest.raises(ValueError, match="two"):
        GeneticSearchEngine.from_checkpoint(
            replace(checkpoint, population=("0",)),
            fitness=_fitness,
            clock=LogicalClock(),
        )


def test_genetic_search_rejects_invalid_population_and_fitness() -> None:
    with pytest.raises(ValueError, match="two"):
        GeneticSearchEngine(("0",), _zero_fitness, generations=1)
    engine = GeneticSearchEngine(("0", "1"), _invalid_fitness, generations=1)
    with pytest.raises(ValueError, match="finite"):
        engine.step()
