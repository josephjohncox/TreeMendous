"""Independent Cartesian-product trial oracle."""

import itertools
from collections.abc import Mapping, Sequence
from typing import Any


def expected_trials(space: Mapping[str, Sequence[Any]]) -> tuple[tuple[tuple[str, Any], ...], ...]:
    names = sorted(space)
    return tuple(tuple(zip(names, values, strict=True)) for values in itertools.product(*(space[name] for name in names)))
