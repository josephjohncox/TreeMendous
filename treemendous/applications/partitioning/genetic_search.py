"""Seeded checkpointable genetic search over fixed-width bit strings."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, cast

from treemendous.applications._shared.claiming import ClaimUnavailableError
from treemendous.applications._shared.clock import Clock
from treemendous.applications.partitioning._runtime import (
    PartitionRuntime,
    RuntimeCheckpoint,
    positive,
)

Fitness = Callable[[str], float]


def _finite_float(value: object, name: str) -> float:
    try:
        converted = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


def _count_ones(value: str) -> float:
    return value.count("1") * 1.0


@dataclass(frozen=True)
class GeneticGeneration:
    """One generation's ordered population and fitness ranking."""

    number: int
    population: tuple[str, ...]
    ranking: tuple[tuple[float, str], ...]


@dataclass(frozen=True)
class GeneticCheckpoint:
    """Replayable process-local genetic state, including PRNG state."""

    generation: int
    generations: int
    mutation_rate: float
    population: tuple[str, ...]
    history: tuple[GeneticGeneration, ...]
    random_state: object
    runtime: RuntimeCheckpoint


class GeneticSearchEngine:
    """Run deterministic selection, crossover, and mutation generations.

    The injected fitness function executes locally and is not serialized. A
    multi-process service must distribute it and durably fence generation
    commits; this engine only supplies a checkpointable in-memory lineage.
    """

    def __init__(
        self,
        population: Sequence[str],
        fitness: Fitness,
        *,
        generations: int,
        seed: int = 0,
        mutation_rate: float = 0.05,
        clock: Clock | None = None,
    ) -> None:
        if isinstance(population, (str, bytes)) or not isinstance(population, Sequence):
            raise TypeError("population must be a sequence of bit strings")
        if len(population) < 2:
            raise ValueError("population must contain at least two candidates")
        width = len(population[0])
        if width == 0 or any(
            not isinstance(item, str)
            or len(item) != width
            or set(item) - {"0", "1"}
            for item in population
        ):
            raise ValueError("population must contain equal-width nonempty bit strings")
        if not callable(fitness):
            raise TypeError("fitness must be callable")
        positive(generations, "generations")
        if type(seed) is not int:
            raise TypeError("seed must be an integer")
        if not isinstance(mutation_rate, (int, float)) or isinstance(mutation_rate, bool):
            raise TypeError("mutation_rate must be numeric")
        converted_rate = _finite_float(mutation_rate, "mutation_rate")
        if not 0.0 <= converted_rate <= 1.0:
            raise ValueError("mutation_rate must be between zero and one")
        self._population = tuple(population)
        self._fitness = fitness
        self._generations = generations
        self._generation = 0
        self._mutation_rate = converted_rate
        self._random = random.Random(seed)
        self._history: list[GeneticGeneration] = []
        self._runtime = PartitionRuntime(generations, clock=clock)

    @staticmethod
    def _score(fitness: Fitness, population: tuple[str, ...]) -> tuple[tuple[float, str], ...]:
        scored: list[tuple[float, str]] = []
        for candidate in population:
            try:
                raw_score = fitness(candidate)
            except (Exception,) as exc:
                raise ValueError("fitness evaluation failed") from exc
            score = _finite_float(raw_score, "fitness value")
            scored.append((score, candidate))
        return tuple(sorted(scored, key=lambda item: (-item[0], item[1])))

    def step(self, *, owner: str = "local") -> GeneticGeneration:
        """Claim and execute exactly one generation."""
        if self._generation >= self._generations:
            raise ClaimUnavailableError("all generations are complete")
        claim = self._runtime.claim(owner, 1)
        prior_random_state = self._random.getstate()
        try:
            ranking = self._score(self._fitness, self._population)
            record = GeneticGeneration(self._generation, self._population, ranking)
            survivors = tuple(
                item[1] for item in ranking[: max(2, len(ranking) // 2)]
            )
            children: list[str] = []
            while len(children) < len(self._population):
                left = survivors[self._random.randrange(len(survivors))]
                right = survivors[self._random.randrange(len(survivors))]
                point = self._random.randrange(1, len(left)) if len(left) > 1 else 1
                child = list(left[:point] + right[point:])
                for index, bit in enumerate(child):
                    if self._random.random() < self._mutation_rate:
                        child[index] = "1" if bit == "0" else "0"
                children.append("".join(child))
        except (Exception,):
            self._random.setstate(prior_random_state)
            self._runtime.abandon(claim)
            raise
        self._population = tuple(children)
        self._history.append(record)
        self._generation += 1
        self._runtime.complete(claim, "generation", {"generation": record.number})
        return record

    def run(self) -> tuple[GeneticGeneration, ...]:
        """Run all remaining generations."""
        while self._generation < self._generations:
            self.step()
        return tuple(self._history)

    def best(self) -> tuple[float, str]:
        """Return the best candidate in the current population."""
        return self._score(self._fitness, self._population)[0]

    def snapshot(self) -> GeneticCheckpoint:
        """Capture complete deterministic local state."""
        return GeneticCheckpoint(
            self._generation,
            self._generations,
            self._mutation_rate,
            self._population,
            tuple(self._history),
            self._random.getstate(),
            self._runtime.checkpoint(),
        )

    checkpoint = snapshot

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: GeneticCheckpoint,
        *,
        fitness: Fitness,
        clock: Clock,
    ) -> GeneticSearchEngine:
        """Restore a generation lineage with its caller-supplied fitness code."""
        if not isinstance(checkpoint, GeneticCheckpoint):
            raise TypeError("checkpoint must be a GeneticCheckpoint")
        if not callable(fitness):
            raise TypeError("fitness must be callable")
        if not 0 <= checkpoint.generation <= checkpoint.generations:
            raise ValueError("checkpoint generation is inconsistent")
        if len(checkpoint.history) != checkpoint.generation:
            raise ValueError("checkpoint history is inconsistent")
        candidate = cls.__new__(cls)
        candidate._population = checkpoint.population
        candidate._fitness = fitness
        candidate._generations = checkpoint.generations
        candidate._generation = checkpoint.generation
        candidate._mutation_rate = checkpoint.mutation_rate
        candidate._random = random.Random()
        candidate._random.setstate(
            cast(tuple[Any, ...], checkpoint.random_state)
        )
        candidate._history = list(checkpoint.history)
        candidate._runtime = PartitionRuntime.from_checkpoint(
            checkpoint.runtime, clock=clock
        )
        return candidate


def create_genetic_search(
    population: Sequence[str] = ("0000", "0011", "1100", "1111"),
    fitness: Fitness | None = None,
    *,
    generations: int = 4,
    seed: int = 0,
    mutation_rate: float = 0.05,
    clock: Clock | None = None,
) -> GeneticSearchEngine:
    """Create a seeded genetic-search job."""
    selected = _count_ones if fitness is None else fitness
    return GeneticSearchEngine(
        population,
        selected,
        generations=generations,
        seed=seed,
        mutation_rate=mutation_rate,
        clock=clock,
    )
