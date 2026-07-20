#!/usr/bin/env python3
"""Run seeded genetic search from any working directory."""

from treemendous.applications.partitioning.genetic_search import GeneticSearchEngine


def fitness(value: str) -> float:
    return value.count("1") * 1.0


def main() -> None:
    engine = GeneticSearchEngine(
        ("000", "001", "110", "111"), fitness, generations=3, seed=5
    )
    history = engine.run()
    if len(history) != 3:
        raise RuntimeError("unexpected genetic history")
    print(f"genetic-search: best={engine.best()[1]}")


if __name__ == "__main__":
    main()
