"""Genetic-search engine contracts."""

import random
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


def test_seeded_generation_matches_fixed_evolution_vector() -> None:
    engine = GeneticSearchEngine(
        ("000", "001", "110", "111"),
        _fitness,
        generations=2,
        seed=7,
        mutation_rate=0.2,
    )

    first = engine.step()
    first_expected = GeneticGeneration(
        0,
        ("000", "001", "110", "111"),
        ((3.0, "111"), (2.0, "110"), (1.0, "001"), (0.0, "000")),
    )
    assert first == first_expected
    first_population = ("101", "010", "100", "100")
    assert engine.snapshot().population == first_population

    second = engine.step()
    second_expected = GeneticGeneration(
        1,
        first_population,
        ((2.0, "101"), (1.0, "010"), (1.0, "100"), (1.0, "100")),
    )
    assert second == second_expected
    final_population = ("010", "100", "101", "000")
    checkpoint = engine.snapshot()
    assert checkpoint.population == final_population
    randomizer = random.Random()
    randomizer.setstate(checkpoint.random_state)  # type: ignore[arg-type]
    assert randomizer.random() == pytest.approx(0.08185501079576984)


def test_restore_rejects_forged_population_transitions_and_random_state() -> None:
    engine = GeneticSearchEngine(
        ("000", "001", "110", "111"),
        _fitness,
        generations=2,
        seed=7,
        mutation_rate=0.2,
    )
    engine.run()
    checkpoint = engine.checkpoint()
    forged_population = ("000", "000", "000", "000")

    with pytest.raises(ValueError, match="final population"):
        GeneticSearchEngine.from_checkpoint(
            replace(checkpoint, population=forged_population),
            fitness=_fitness,
            clock=LogicalClock(),
        )

    forged_record = replace(
        checkpoint.history[1],
        population=forged_population,
        ranking=expected_ranking(forged_population, _fitness),
    )
    with pytest.raises(ValueError, match="population transition"):
        GeneticSearchEngine.from_checkpoint(
            replace(checkpoint, history=(checkpoint.history[0], forged_record)),
            fitness=_fitness,
            clock=LogicalClock(),
        )

    with pytest.raises(ValueError, match="random state contradicts"):
        GeneticSearchEngine.from_checkpoint(
            replace(checkpoint, random_state=random.Random(999).getstate()),
            fitness=_fitness,
            clock=LogicalClock(),
        )


def test_restore_requires_generation_application_event_kind() -> None:
    engine = GeneticSearchEngine(("00", "11"), _fitness, generations=1)
    engine.run()
    checkpoint = engine.checkpoint()
    event = checkpoint.runtime.events.events[0]
    request = checkpoint.runtime.events.requests[0]
    forged_runtime = replace(
        checkpoint.runtime,
        events=replace(
            checkpoint.runtime.events,
            events=(
                replace(
                    event,
                    kind="not-a-generation",
                    idempotency_key="not-a-generation",
                ),
            ),
            requests=(
                replace(
                    request,
                    key="not-a-generation",
                    kind="not-a-generation",
                ),
            ),
        ),
    )

    with pytest.raises(ClaimInvariantError, match="event kind"):
        GeneticSearchEngine.from_checkpoint(
            replace(checkpoint, runtime=forged_runtime),
            fitness=_fitness,
            clock=LogicalClock(),
        )


def test_restore_rejects_contradictory_application_and_runtime_progress() -> None:
    population = ("00", "11")
    engine = GeneticSearchEngine(population, _fitness, generations=2)
    engine.step()
    checkpoint = engine.checkpoint()
    empty_runtime = (
        GeneticSearchEngine(population, _fitness, generations=2).checkpoint().runtime
    )
    mixed = replace(checkpoint, runtime=empty_runtime)

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
