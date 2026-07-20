"""Finite point-set differential evidence for optimized box indexes."""

from __future__ import annotations

from collections.abc import Callable
from random import Random

import pytest

from tests.oracles.multidimensional.brute_box_index import BruteBoxIndex
from treemendous.multidimensional import (
    BoundedBoxIndex,
    Box,
    BoxIndex2D,
    BoxIndex3D,
    BoxIndex4D,
    BoxIndexProtocol,
)


def _bounds(random: Random, dimensions: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    lower = tuple(random.randint(-3, 3) for _ in range(dimensions))
    upper = tuple(value + random.randint(1, 3) for value in lower)
    return lower, upper


@pytest.mark.parametrize(
    "dimensions,factory",
    [
        (2, BoxIndex2D),
        (3, BoxIndex3D),
        (4, BoxIndex4D),
    ],
)
def test_fixed_projection_matches_independent_point_oracle(
    dimensions: int,
    factory: Callable[[], BoxIndexProtocol],
) -> None:
    random = Random(9100 + dimensions)
    index = factory()
    oracle = BruteBoxIndex(dimensions)
    handles = []
    oracle_handles = []
    for ordinal in range(35):
        lower, upper = _bounds(random, dimensions)
        handles.append(index.insert(Box(lower, upper), ordinal))
        oracle_handles.append(oracle.insert(lower, upper, ordinal))

    # Exercise index maintenance, including geometry updates and holes in the
    # monotonic handle sequence, before comparing a separate query corpus.
    for position in (2, 9, 17):
        lower, upper = _bounds(random, dimensions)
        index.update(handles[position], box=Box(lower, upper), data=100 + position)
        oracle.update(
            oracle_handles[position],
            lower=lower,
            upper=upper,
            data=100 + position,
        )
    for position in reversed((4, 12, 23)):
        index.remove(handles[position])
        oracle.remove(oracle_handles[position])

    for _ in range(80):
        lower, upper = _bounds(random, dimensions)
        observed = [entry.data for entry in index.overlaps(Box(lower, upper))]
        expected = [entry.data for entry in oracle.overlaps(lower, upper)]
        assert observed == expected


@pytest.mark.parametrize("dimensions", [2, 3, 4, 6, 8])
def test_bounded_nd_grid_matches_independent_point_oracle(dimensions: int) -> None:
    random = Random(12000 + dimensions)
    bounds = Box((-4,) * dimensions, (5,) * dimensions)
    index = BoundedBoxIndex(
        bounds,
        (2,) * dimensions,
        max_total_cells=400_000,
        max_cells_per_entry=10_000,
        max_cells_per_query=10_000,
        max_total_postings=200_000,
    )
    oracle = BruteBoxIndex(dimensions)
    handles = []
    oracle_handles = []
    for ordinal in range(20):
        lower = tuple(random.randint(-3, 2) for _ in range(dimensions))
        upper = tuple(value + random.randint(1, 2) for value in lower)
        handles.append(index.insert(Box(lower, upper), ordinal))
        oracle_handles.append(oracle.insert(lower, upper, ordinal))

    replacement_lower = (-1,) * dimensions
    replacement_upper = (1,) * dimensions
    index.update(
        handles[3],
        box=Box(replacement_lower, replacement_upper),
        data="updated",
    )
    oracle.update(
        oracle_handles[3],
        lower=replacement_lower,
        upper=replacement_upper,
        data="updated",
    )
    index.remove(handles[7])
    oracle.remove(oracle_handles[7])

    for _ in range(40):
        lower = tuple(random.randint(-4, 3) for _ in range(dimensions))
        upper = tuple(value + random.randint(1, min(2, 5 - value)) for value in lower)
        observed = [entry.data for entry in index.overlaps(Box(lower, upper))]
        expected = [entry.data for entry in oracle.overlaps(lower, upper)]
        assert observed == expected


def test_adversarial_duplicates_faces_huge_boxes_and_skew() -> None:
    huge = 10**100
    index = BoxIndex4D()
    boxes = [
        Box((-huge, -1, -1, -1), (huge, 1, 1, 1)),
        Box((-huge, -1, -1, -1), (huge, 1, 1, 1)),
        Box((huge, -1, -1, -1), (huge + 1, 1, 1, 1)),
        Box((-2, 1000, -2, -2), (2, 1001, 2, 2)),
    ]
    handles = [index.insert(box, ordinal) for ordinal, box in enumerate(boxes)]

    matches = index.overlaps(Box((-1, -1, -1, -1), (1, 1, 1, 1)))
    assert [entry.handle for entry in matches] == handles[:2]
    separated = index.overlaps(Box((huge + 1, -1, -1, -1), (huge + 2, 1, 1, 1)))
    assert len(separated) == 0
