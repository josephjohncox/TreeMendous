"""Structurally independent evidence for genetic-search scenarios."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

Fitness = Callable[[str], float]
GenerationSpec = tuple[int, tuple[str, ...], tuple[tuple[float, str], ...]]


def expected_ranking(
    population: Sequence[str], fitness: Fitness
) -> tuple[tuple[float, str], ...]:
    """Rank a population independently by score and lexical tie breaker."""
    return tuple(
        sorted(
            ((fitness(item), item) for item in population),
            key=lambda item: (-item[0], item[1]),
        )
    )


def expected_invariant_search(
    candidate: str,
    *,
    population_size: int,
    generations: int,
    score: float,
    seed: int,
) -> tuple[tuple[GenerationSpec, ...], tuple[str, ...], object]:
    """Derive a zero-mutation invariant workload without simulating evolution."""
    population = (candidate,) * population_size
    ranking = ((score, candidate),) * population_size
    history = tuple(
        (generation, population, ranking) for generation in range(generations)
    )
    initial_random_state = random.Random(seed).getstate()
    return history, population, initial_random_state
