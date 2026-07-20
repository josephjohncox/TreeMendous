"""Correctness-checked smoke workload for Cartesian parameter search."""

from collections.abc import Mapping
from typing import Any

from tests.oracles.applications.partitioning.hyperparameter_search import (
    expected_trials,
)
from treemendous.applications.partitioning.hyperparameter_search import (
    HyperparameterSearchEngine,
)


def _objective(parameters: Mapping[str, Any]) -> float:
    return parameters["a"] * 10.0 + parameters["b"]


def run_smoke() -> int:
    space = {"b": tuple(range(10)), "a": tuple(range(20))}
    engine = HyperparameterSearchEngine(space, _objective)
    expected = expected_trials(space)
    if tuple(engine.parameters_for(i) for i in range(len(expected))) != expected:
        raise AssertionError("trial ID mapping differs from Cartesian oracle")
    ranking = engine.run(shard_size=17)
    if len(ranking) != len(expected):
        raise AssertionError("not all hyperparameter trials were ranked")
    return len(ranking)
