"""Independent Cartesian-product trial and ranking oracle."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

ParameterTuple = tuple[tuple[str, Any], ...]
TrialSpec = tuple[int, ParameterTuple, float]


def expected_trials(
    space: Mapping[str, Sequence[Any]],
) -> tuple[ParameterTuple, ...]:
    """Enumerate the stable Cartesian trial-ID mapping."""
    names = sorted(space)
    return tuple(
        tuple(zip(names, values, strict=True))
        for values in itertools.product(*(space[name] for name in names))
    )


def expected_ranking(
    space: Mapping[str, Sequence[Any]],
    objective: Callable[[Mapping[str, Any]], float],
    *,
    maximize: bool = True,
) -> tuple[TrialSpec, ...]:
    """Evaluate and rank every Cartesian trial without using the engine."""
    trials = tuple(
        (trial_id, parameters, float(objective(dict(parameters))))
        for trial_id, parameters in enumerate(expected_trials(space))
    )
    direction = -1.0 if maximize else 1.0
    return tuple(sorted(trials, key=lambda item: (direction * item[2], item[0])))
