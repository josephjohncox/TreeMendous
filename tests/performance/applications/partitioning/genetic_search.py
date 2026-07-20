"""Attested benchmark for seeded genetic search."""

from __future__ import annotations

from tests.oracles.applications.partitioning.genetic_search import expected_search
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from tests.performance.applications.partitioning._common import validate_case
from treemendous.applications.partitioning.genetic_search import (
    GeneticGeneration,
    GeneticSearchEngine,
)

_DEFAULT_OPERATIONS = 12
_MAX_OPERATIONS = 64
_DEFAULT_SEED = 19
_MUTATION_RATE = 0.075


def _fitness(value: str) -> float:
    return value.count("1") * 2.0 + value.count("11") * 0.25


def _generation_tuple(
    generation: GeneticGeneration,
) -> tuple[int, tuple[str, ...], tuple[tuple[float, str], ...]]:
    return generation.number, generation.population, generation.ranking


def run_benchmark(
    operations: int = _DEFAULT_OPERATIONS, seed: int = _DEFAULT_SEED
) -> ApplicationSample:
    """Run and attest every generation of a bounded seeded search."""
    validate_case(operations, seed, maximum=_MAX_OPERATIONS)
    population = tuple(f"{(value * 37 + seed) % 256:08b}" for value in range(24))
    engine = GeneticSearchEngine(
        population,
        _fitness,
        generations=operations,
        seed=seed,
        mutation_rate=_MUTATION_RATE,
    )

    def execute() -> tuple[GeneticGeneration, ...]:
        return engine.run()

    def observe(raw: tuple[GeneticGeneration, ...]) -> ApplicationOutcome:
        checkpoint = engine.snapshot()
        history = tuple(_generation_tuple(item) for item in raw)
        state_history = tuple(_generation_tuple(item) for item in checkpoint.history)
        return ApplicationOutcome(
            results=history,
            final_state={
                "generation": checkpoint.generation,
                "generations": checkpoint.generations,
                "mutation_rate": checkpoint.mutation_rate,
                "population": checkpoint.population,
                "history": state_history,
                "random_state": checkpoint.random_state,
            },
            counters={
                "generations_executed": len(checkpoint.history),
                "fitness_evaluations": len(population) * len(checkpoint.history),
                "population_size": len(checkpoint.population),
                "run_calls": 1,
            },
        )

    def oracle() -> ApplicationOutcome:
        history, final_population, random_state = expected_search(
            population,
            _fitness,
            generations=operations,
            seed=seed,
            mutation_rate=_MUTATION_RATE,
        )
        return ApplicationOutcome(
            results=history,
            final_state={
                "generation": operations,
                "generations": operations,
                "mutation_rate": _MUTATION_RATE,
                "population": final_population,
                "history": history,
                "random_state": random_state,
            },
            counters={
                "generations_executed": operations,
                "fitness_evaluations": len(population) * operations,
                "population_size": len(population),
                "run_calls": 1,
            },
        )

    return run_application_case(
        scenario_id="partitioning.genetic_search",
        operations=operations,
        execute=execute,
        observe=observe,
        oracle=oracle,
    )


def run_smoke() -> int:
    """Run the default attested workload and return its operation count."""
    return run_benchmark().operations
