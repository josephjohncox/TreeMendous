"""Independent seeded simulator for genetic-search benchmark evidence."""

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


def expected_search(
    population: Sequence[str],
    fitness: Fitness,
    *,
    generations: int,
    seed: int,
    mutation_rate: float,
) -> tuple[tuple[GenerationSpec, ...], tuple[str, ...], object]:
    """Simulate selection, crossover, mutation, and final PRNG state."""
    current = tuple(population)
    randomizer = random.Random(seed)
    history: list[GenerationSpec] = []
    for generation in range(generations):
        ranking = expected_ranking(current, fitness)
        history.append((generation, current, ranking))
        survivors = tuple(item[1] for item in ranking[: max(2, len(ranking) // 2)])
        children: list[str] = []
        while len(children) < len(current):
            left = survivors[randomizer.randrange(len(survivors))]
            right = survivors[randomizer.randrange(len(survivors))]
            point = randomizer.randrange(1, len(left)) if len(left) > 1 else 1
            child = list(left[:point] + right[point:])
            for index, bit in enumerate(child):
                if randomizer.random() < mutation_rate:
                    child[index] = "1" if bit == "0" else "0"
            children.append("".join(child))
        current = tuple(children)
    return tuple(history), current, randomizer.getstate()
