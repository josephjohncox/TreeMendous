"""Finite contract tests for half-open multidimensional boxes."""

from __future__ import annotations

import pytest

from treemendous.multidimensional import Box


def test_box_geometry_is_half_open_and_exact() -> None:
    outer = Box((-2, 3, 10), (4, 8, 12))
    inner = Box((0, 4, 10), (2, 7, 12))
    overlapping = Box((3, 7, 11), (6, 9, 14))
    touching = Box((4, 3, 10), (5, 8, 12))

    assert outer.dimensions == 3
    assert outer.volume == 60
    assert outer.contains(inner)
    assert outer.overlaps(overlapping)
    assert overlapping.overlaps(outer)
    assert not outer.overlaps(touching)


def test_box_accepts_arbitrary_size_integers_and_iterables() -> None:
    huge = 10**100
    box = Box(iter((-huge, 0)), iter((huge, 2)))
    expected_lower = (-huge, 0)
    expected_upper = (huge, 2)
    assert box.lower == expected_lower
    assert box.upper == expected_upper
    assert box.volume == 4 * huge


@pytest.mark.parametrize(
    "lower,upper,error",
    [
        ((), (), ValueError),
        ((0,), (1, 2), ValueError),
        ((0, 0), (1, 0), ValueError),
        ((0, 2), (1, 1), ValueError),
        ((False, 0), (1, 1), TypeError),
        ((0.0, 0), (1, 1), TypeError),
        ((0, 0), (True, 1), TypeError),
    ],
)
def test_box_rejects_invalid_bounds(lower, upper, error) -> None:
    with pytest.raises(error):
        Box(lower, upper)


def test_containment_and_coordinate_transform_laws() -> None:
    outer = Box((-3, 1, 5), (8, 9, 12))
    middle = Box((-1, 2, 6), (6, 8, 11))
    inner = Box((0, 3, 7), (4, 7, 10))
    assert outer.contains(outer)
    assert outer.contains(middle)
    assert middle.contains(inner)
    assert outer.contains(inner)

    offset = (10, -4, 3)
    translated = Box(
        tuple(value + delta for value, delta in zip(outer.lower, offset, strict=True)),
        tuple(value + delta for value, delta in zip(outer.upper, offset, strict=True)),
    )
    permuted = Box(tuple(reversed(outer.lower)), tuple(reversed(outer.upper)))
    assert translated.volume == outer.volume
    assert permuted.volume == outer.volume


def test_face_edge_and_corner_contact_are_disjoint() -> None:
    source = Box((0, 0, 0), (2, 2, 2))
    for touching in (
        Box((2, 0, 0), (3, 2, 2)),
        Box((2, 2, 0), (3, 3, 2)),
        Box((2, 2, 2), (3, 3, 3)),
    ):
        assert not source.overlaps(touching)
        assert not touching.overlaps(source)


def test_cross_dimensional_predicates_are_rejected() -> None:
    line = Box((0,), (1,))
    plane = Box((0, 0), (1, 1))
    with pytest.raises(ValueError, match="matching dimensions"):
        line.overlaps(plane)
    with pytest.raises(ValueError, match="matching dimensions"):
        line.contains(plane)
