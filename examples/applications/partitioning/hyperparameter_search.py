#!/usr/bin/env python3
"""Run Cartesian hyperparameter search from any working directory."""

from collections.abc import Mapping
from typing import Any

from treemendous.applications.partitioning.hyperparameter_search import (
    HyperparameterSearchEngine,
)


def objective(parameters: Mapping[str, Any]) -> float:
    return parameters["depth"] - parameters["rate"]


def main() -> None:
    engine = HyperparameterSearchEngine(
        {"rate": (0.1, 0.2), "depth": (2, 4)}, objective
    )
    best = engine.run(shard_size=2)[0]
    if best.trial_id != 2:
        raise RuntimeError("unexpected hyperparameter rank")
    print(f"hyperparameter-search: best trial {best.trial_id}")


if __name__ == "__main__":
    main()
