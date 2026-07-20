"""Hyperparameter-search engine contracts."""

from collections.abc import Mapping
from math import inf
from typing import Any

import pytest

from tests.oracles.applications.partitioning.hyperparameter_search import (
    expected_trials,
)
from treemendous.applications.partitioning.hyperparameter_search import (
    HyperparameterSearchEngine,
)


def _objective(parameters: Mapping[str, Any]) -> float:
    return parameters["depth"] - parameters["rate"]


def _zero(_: Mapping[str, Any]) -> float:
    return 0.0


def _infinite(_: Mapping[str, Any]) -> float:
    return inf


def test_cartesian_ids_and_objective_ranking_are_deterministic() -> None:
    space = {"rate": (0.1, 0.2), "depth": (2, 4)}
    engine = HyperparameterSearchEngine(space, _objective)
    expected = expected_trials(space)
    assert tuple(engine.parameters_for(index) for index in range(4)) == expected
    ranking = engine.run(shard_size=1)
    expected_ranking = (2, 3, 0, 1)
    assert tuple(item.trial_id for item in ranking) == expected_ranking


def test_hyperparameter_search_rejects_empty_axes_and_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="empty"):
        HyperparameterSearchEngine({"x": ()}, _zero)
    engine = HyperparameterSearchEngine({"x": (1,)}, _infinite)
    with pytest.raises(ValueError, match="finite"):
        engine.run()
