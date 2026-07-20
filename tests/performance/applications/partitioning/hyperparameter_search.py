"""Attested benchmark for Cartesian hyperparameter search."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tests.oracles.applications.partitioning.hyperparameter_search import (
    expected_ranking,
)
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from tests.performance.applications.partitioning._common import validate_case
from treemendous.applications.partitioning.hyperparameter_search import (
    HyperparameterSearchEngine,
    TrialResult,
)

_DEFAULT_OPERATIONS = 160
_MAX_OPERATIONS = 1_000
_DEFAULT_SEED = 31
_SHARD_SIZE = 17


def _objective(parameters: Mapping[str, Any]) -> float:
    return float(parameters["candidate"] * 10 + parameters["offset"])


def _trial_tuple(
    trial: TrialResult,
) -> tuple[int, tuple[tuple[str, Any], ...], float]:
    return trial.trial_id, trial.parameters, trial.score


def run_benchmark(
    operations: int = _DEFAULT_OPERATIONS, seed: int = _DEFAULT_SEED
) -> ApplicationSample:
    """Evaluate and attest an exactly sized Cartesian trial grid."""
    validate_case(operations, seed, maximum=_MAX_OPERATIONS)
    space = {
        "offset": (seed % 23,),
        "candidate": tuple(range(operations)),
    }
    engine = HyperparameterSearchEngine(space, _objective)

    def execute() -> tuple[TrialResult, ...]:
        return engine.run(shard_size=_SHARD_SIZE)

    def observe(raw: tuple[TrialResult, ...]) -> ApplicationOutcome:
        snapshot = engine.snapshot()
        ranking = tuple(_trial_tuple(item) for item in raw)
        state_ranking = tuple(_trial_tuple(item) for item in snapshot.ranking)
        return ApplicationOutcome(
            results=ranking,
            final_state={
                "parameter_names": snapshot.parameter_names,
                "ranking": state_ranking,
            },
            counters={
                "trials_evaluated": len(snapshot.ranking),
                "evaluation_bands": (operations + _SHARD_SIZE - 1) // _SHARD_SIZE,
                "run_calls": 1,
            },
        )

    def oracle() -> ApplicationOutcome:
        ranking = expected_ranking(space, _objective)
        return ApplicationOutcome(
            results=ranking,
            final_state={
                "parameter_names": tuple(sorted(space)),
                "ranking": ranking,
            },
            counters={
                "trials_evaluated": len(ranking),
                "evaluation_bands": (operations + _SHARD_SIZE - 1) // _SHARD_SIZE,
                "run_calls": 1,
            },
        )

    return run_application_case(
        scenario_id="partitioning.hyperparameter_search",
        operations=operations,
        execute=execute,
        observe=observe,
        oracle=oracle,
    )


def run_smoke() -> int:
    """Run the default attested workload and return its operation count."""
    return run_benchmark().operations
