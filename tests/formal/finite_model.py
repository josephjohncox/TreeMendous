"""Bounded integer box universe used by executable refinement checks."""

from __future__ import annotations

from itertools import product

Bounds = tuple[tuple[int, ...], tuple[int, ...]]


def finite_boxes(*, dimensions: int, extent: int) -> tuple[Bounds, ...]:
    """Enumerate every nonempty box in ``range(extent + 1)^dimensions``."""
    intervals = tuple(
        (lower, upper)
        for lower in range(extent)
        for upper in range(lower + 1, extent + 1)
    )
    return tuple(
        (
            tuple(interval[0] for interval in axes),
            tuple(interval[1] for interval in axes),
        )
        for axes in product(intervals, repeat=dimensions)
    )
