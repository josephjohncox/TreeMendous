"""Seeded checkpointable genetic search over fixed-width bit strings."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, cast

from treemendous.applications._shared.claiming import (
    ClaimInvariantError,
    ClaimState,
    ClaimUnavailableError,
)
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


def _validated_population(population: object) -> tuple[str, ...]:
    if isinstance(population, (str, bytes)) or not isinstance(population, Sequence):
        raise TypeError("population must be a sequence of bit strings")
    if len(population) < 2:
        raise ValueError("population must contain at least two candidates")
    width = len(population[0])
    if width == 0 or any(
        not isinstance(item, str) or len(item) != width or set(item) - {"0", "1"}
        for item in population
    ):
        raise ValueError("population must contain equal-width nonempty bit strings")
    return tuple(population)


def _validated_rate(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("mutation_rate must be numeric")
    converted = _finite_float(value, "mutation_rate")
    if not 0.0 <= converted <= 1.0:
        raise ValueError("mutation_rate must be between zero and one")
    return converted


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
    initial_random_state: object
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
        checked_population = _validated_population(population)
        if not callable(fitness):
            raise TypeError("fitness must be callable")
        positive(generations, "generations")
        if type(seed) is not int:
            raise TypeError("seed must be an integer")
        converted_rate = _validated_rate(mutation_rate)
        self._population = checked_population
        self._fitness = fitness
        self._generations = generations
        self._generation = 0
        self._mutation_rate = converted_rate
        self._random = random.Random(seed)
        self._initial_random_state = self._random.getstate()
        self._history: list[GeneticGeneration] = []
        self._runtime = PartitionRuntime(generations, clock=clock)

    @staticmethod
    def _score(
        fitness: Fitness, population: tuple[str, ...]
    ) -> tuple[tuple[float, str], ...]:
        scored: list[tuple[float, str]] = []
        for candidate in population:
            try:
                raw_score = fitness(candidate)
            except (Exception,) as exc:
                raise ValueError("fitness evaluation failed") from exc
            score = _finite_float(raw_score, "fitness value")
            scored.append((score, candidate))
        return tuple(sorted(scored, key=lambda item: (-item[0], item[1])))

    @staticmethod
    def _next_population(
        population: tuple[str, ...],
        ranking: tuple[tuple[float, str], ...],
        mutation_rate: float,
        randomizer: random.Random,
    ) -> tuple[str, ...]:
        survivors = tuple(item[1] for item in ranking[: max(2, len(ranking) // 2)])
        if mutation_rate == 0.0 and len(set(survivors)) == 1:
            return (survivors[0],) * len(population)
        children: list[str] = []
        while len(children) < len(population):
            left = survivors[randomizer.randrange(len(survivors))]
            right = survivors[randomizer.randrange(len(survivors))]
            point = randomizer.randrange(1, len(left)) if len(left) > 1 else 1
            child = list(left[:point] + right[point:])
            for index, bit in enumerate(child):
                if randomizer.random() < mutation_rate:
                    child[index] = "1" if bit == "0" else "0"
            children.append("".join(child))
        return tuple(children)

    def step(self, *, owner: str = "local") -> GeneticGeneration:
        """Claim and execute exactly one generation."""
        if self._generation >= self._generations:
            raise ClaimUnavailableError("all generations are complete")
        claim = self._runtime.claim(owner, 1)

        def prepare() -> tuple[
            GeneticGeneration, tuple[str, ...], object, list[GeneticGeneration]
        ]:
            randomizer = random.Random()
            randomizer.setstate(self._random.getstate())
            ranking = self._score(self._fitness, self._population)
            record = GeneticGeneration(self._generation, self._population, ranking)
            next_population = self._next_population(
                self._population, ranking, self._mutation_rate, randomizer
            )
            return (
                record,
                next_population,
                randomizer.getstate(),
                [*self._history, record],
            )

        def commit(
            value: tuple[
                GeneticGeneration,
                tuple[str, ...],
                object,
                list[GeneticGeneration],
            ],
        ) -> None:
            self._population = value[1]
            self._random.setstate(cast(tuple[Any, ...], value[2]))
            self._history = value[3]
            self._generation += 1

        prepared = self._runtime.execute_claim(
            claim,
            kind="generation",
            prepare=prepare,
            commit=commit,
            result=lambda value: {"generation": value[0].number},
        )
        return prepared[0]

    def run(self) -> tuple[GeneticGeneration, ...]:
        """Run all remaining generations."""
        while self._generation < self._generations:
            self.step()
        return tuple(self._history)

    def best(self) -> tuple[float, str]:
        """Return the best candidate in the current population."""
        return self._score(self._fitness, self._population)[0]

    def _snapshot(self) -> GeneticCheckpoint:
        return GeneticCheckpoint(
            self._generation,
            self._generations,
            self._mutation_rate,
            self._population,
            tuple(self._history),
            self._random.getstate(),
            self._initial_random_state,
            self._runtime.checkpoint(),
        )

    def snapshot(self) -> GeneticCheckpoint:
        """Capture complete deterministic local state."""
        return self._runtime.observe(self._snapshot)

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
        positive(checkpoint.generations, "checkpoint generations")
        if type(checkpoint.generation) is not int or not (
            0 <= checkpoint.generation <= checkpoint.generations
        ):
            raise ValueError("checkpoint generation is inconsistent")
        population = _validated_population(checkpoint.population)
        mutation_rate = _validated_rate(checkpoint.mutation_rate)
        if not isinstance(checkpoint.history, tuple) or (
            len(checkpoint.history) != checkpoint.generation
        ):
            raise ValueError("checkpoint history is inconsistent")
        for number, record in enumerate(checkpoint.history):
            if not isinstance(record, GeneticGeneration) or record.number != number:
                raise ValueError("checkpoint history numbering is inconsistent")
            historical_population = _validated_population(record.population)
            if (
                len(historical_population) != len(population)
                or len(historical_population[0]) != len(population[0])
                or record.ranking != cls._score(fitness, historical_population)
            ):
                raise ValueError("checkpoint history population is inconsistent")

        replay_randomizer = random.Random()
        randomizer = random.Random()
        try:
            replay_randomizer.setstate(
                cast(tuple[Any, ...], checkpoint.initial_random_state)
            )
            randomizer.setstate(cast(tuple[Any, ...], checkpoint.random_state))
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint random state is invalid") from exc

        replayed_population = (
            checkpoint.history[0].population if checkpoint.history else population
        )
        for record in checkpoint.history:
            if record.population != replayed_population:
                raise ValueError("checkpoint population transition is inconsistent")
            replayed_population = cls._next_population(
                replayed_population,
                record.ranking,
                mutation_rate,
                replay_randomizer,
            )
        if replayed_population != population:
            raise ValueError("checkpoint final population is inconsistent")
        if replay_randomizer.getstate() != randomizer.getstate():
            raise ValueError("checkpoint random state contradicts population lineage")

        runtime = PartitionRuntime.from_checkpoint(checkpoint.runtime, clock=clock)
        runtime_claims = checkpoint.runtime.claims
        if runtime_claims.domain.measure != checkpoint.generations:
            raise ClaimInvariantError("runtime domain contradicts generation count")
        completed = tuple(
            claim
            for claim in runtime_claims.claims
            if claim.state is ClaimState.COMPLETED
        )
        if any(
            claim.state in {ClaimState.ACTIVE, ClaimState.EXPIRED}
            for claim in runtime_claims.claims
        ):
            raise ClaimInvariantError(
                "checkpoint contains unfinished generation claims"
            )
        completed_by_generation: list[tuple[int, Any]] = []
        for claim in completed:
            result = dict(claim.result)
            generation = result.get("generation")
            if set(result) != {"generation"} or type(generation) is not int:
                raise ClaimInvariantError(
                    "runtime generation completion metadata is inconsistent"
                )
            completed_by_generation.append((generation, claim))
        completed_by_generation.sort(key=lambda item: item[0])
        if len(completed) != checkpoint.generation or tuple(
            generation for generation, _ in completed_by_generation
        ) != tuple(range(checkpoint.generation)):
            raise ClaimInvariantError("runtime completion count contradicts generation")
        if any(
            claim.span.length != 1 or claim.span.start != generation
            for generation, claim in completed_by_generation
        ):
            raise ClaimInvariantError("runtime generation claims are inconsistent")
        events_by_stream = {
            event.stream: event for event in checkpoint.runtime.events.events
        }
        generation_events = tuple(
            events_by_stream[f"work:{claim.claim_id}"]
            for _, claim in completed_by_generation
        )
        if any(event.kind != "generation" for event in generation_events):
            raise ClaimInvariantError("runtime generation event kind is inconsistent")
        if tuple(event.sequence for event in generation_events) != tuple(
            range(1, checkpoint.generation + 1)
        ):
            raise ClaimInvariantError("runtime generation event order is inconsistent")

        candidate = cls.__new__(cls)
        candidate._population = population
        candidate._fitness = fitness
        candidate._generations = checkpoint.generations
        candidate._generation = checkpoint.generation
        candidate._mutation_rate = mutation_rate
        candidate._random = randomizer
        candidate._initial_random_state = cast(
            tuple[Any, ...], checkpoint.initial_random_state
        )
        candidate._history = list(checkpoint.history)
        candidate._runtime = runtime
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
