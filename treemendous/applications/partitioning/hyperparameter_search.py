"""Cartesian-product hyperparameter trials with deterministic objective ranking."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any

from treemendous.applications._shared.claiming import ClaimUnavailableError, WorkClaim
from treemendous.applications._shared.clock import Clock
from treemendous.applications.partitioning._runtime import (
    PartitionRuntime,
    nonempty,
    positive,
)

Objective = Callable[[Mapping[str, Any]], float]


@dataclass(frozen=True)
class TrialResult:
    """One trial's Cartesian ordinal, parameters, and finite objective."""

    trial_id: int
    parameters: tuple[tuple[str, Any], ...]
    score: float


@dataclass(frozen=True)
class HyperparameterSnapshot:
    """Immutable trial results in objective rank order."""

    parameter_names: tuple[str, ...]
    ranking: tuple[TrialResult, ...]


class HyperparameterSearchEngine:
    """Map trial IDs to a stable Cartesian grid and rank objective results.

    The objective and result map are local Python objects. Distributed callers
    must persist the grid and enforce fencing tokens when committing scores.
    """

    def __init__(
        self,
        space: Mapping[str, Sequence[Any]],
        objective: Objective,
        *,
        maximize: bool = True,
        clock: Clock | None = None,
    ) -> None:
        if not isinstance(space, Mapping) or not space:
            raise ValueError("space must be a nonempty mapping")
        if not callable(objective):
            raise TypeError("objective must be callable")
        if type(maximize) is not bool:
            raise TypeError("maximize must be a boolean")
        normalized: list[tuple[str, tuple[Any, ...]]] = []
        for name, values in space.items():
            nonempty(name, "parameter name")
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise TypeError("parameter values must be a sequence")
            if not values:
                raise ValueError("parameter values must not be empty")
            normalized.append((name, tuple(deepcopy(value) for value in values)))
        normalized.sort(key=lambda item: item[0])
        self._names = tuple(item[0] for item in normalized)
        self._trials = tuple(itertools.product(*(item[1] for item in normalized)))
        self._objective = objective
        self._maximize = maximize
        self._results: dict[int, TrialResult] = {}
        self._runtime = PartitionRuntime(len(self._trials), clock=clock)

    def parameters_for(self, trial_id: int) -> tuple[tuple[str, Any], ...]:
        """Return the stable parameter tuple for one Cartesian ordinal."""
        if type(trial_id) is not int or not 0 <= trial_id < len(self._trials):
            raise ValueError("trial_id is outside the Cartesian grid")
        values = deepcopy(self._trials[trial_id])
        return tuple(zip(self._names, values, strict=True))

    def claim(self, owner: str, length: int) -> WorkClaim:
        """Claim deterministic trial IDs."""
        return self._runtime.claim(owner, length)

    def evaluate_claim(self, claim: WorkClaim) -> tuple[TrialResult, ...]:
        """Evaluate and record all trial IDs in one claimed band."""
        def prepare() -> tuple[tuple[TrialResult, ...], dict[int, TrialResult]]:
            evaluated: list[TrialResult] = []
            results = self._results.copy()
            for trial_id in range(claim.span.start, claim.span.end):
                parameters = self.parameters_for(trial_id)
                try:
                    raw_score = self._objective(deepcopy(dict(parameters)))
                    score = float(raw_score)
                except (Exception,) as exc:
                    raise ValueError(
                        "objective must return a finite number"
                    ) from exc
                if not isfinite(score):
                    raise ValueError("objective must return a finite number")
                trial = TrialResult(trial_id, parameters, score)
                evaluated.append(trial)
                results[trial_id] = trial
            return tuple(evaluated), results

        prepared = self._runtime.execute_claim(
            claim,
            kind="evaluated",
            prepare=prepare,
            commit=lambda value: setattr(self, "_results", value[1]),
            result=lambda value: {"trials": len(value[0])},
        )
        return deepcopy(prepared[0])

    def _ranking(self) -> tuple[TrialResult, ...]:
        direction = -1.0 if self._maximize else 1.0
        ranking = tuple(
            sorted(
                self._results.values(),
                key=lambda item: (direction * item.score, item.trial_id),
            )
        )
        return deepcopy(ranking)

    def ranking(self) -> tuple[TrialResult, ...]:
        """Return score rank with trial ID as deterministic tie breaker."""
        return self._runtime.observe(self._ranking)

    def run(self, *, shard_size: int = 32) -> tuple[TrialResult, ...]:
        """Evaluate all remaining grid trials and return ranking."""
        positive(shard_size, "shard_size")
        while True:
            try:
                claim = self.claim("local", shard_size)
            except ClaimUnavailableError:
                break
            self.evaluate_claim(claim)
        return self.ranking()

    def _snapshot(self) -> HyperparameterSnapshot:
        return HyperparameterSnapshot(self._names, self._ranking())

    def snapshot(self) -> HyperparameterSnapshot:
        """Return detached ranked state."""
        return self._runtime.observe(self._snapshot)

    def audit_snapshot(self) -> tuple[HyperparameterSnapshot, object]:
        """Capture non-restorable application and runtime audit evidence."""
        return self._runtime.audit_snapshot(self._snapshot)


def _default_objective(parameters: Mapping[str, Any]) -> float:
    total = 0.0
    for value in parameters.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += value
    return total


def create_hyperparameter_search(
    space: Mapping[str, Sequence[Any]] | None = None,
    objective: Objective = _default_objective,
    *,
    maximize: bool = True,
    clock: Clock | None = None,
) -> HyperparameterSearchEngine:
    """Create a Cartesian-product search job."""
    selected = {"depth": (2, 4), "rate": (0.1, 0.2)} if space is None else space
    return HyperparameterSearchEngine(
        selected, objective, maximize=maximize, clock=clock
    )
