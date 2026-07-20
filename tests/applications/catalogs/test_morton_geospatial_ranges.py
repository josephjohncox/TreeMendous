"""Morton geospatial candidate/filter contract."""

import pytest

from tests.oracles.applications.catalogs.morton_geospatial_ranges import search
from treemendous.applications.catalogs.morton_geospatial_ranges import (
    MortonGeospatialCatalog,
    create_catalog,
    morton_decompose,
    morton_encode,
)


def test_morton_bit_placement_matches_independent_fixed_vectors() -> None:
    vectors = (
        (0, 0, 3, 0),
        (1, 0, 3, 1),
        (0, 1, 3, 2),
        (1, 1, 3, 3),
        (2, 0, 3, 4),
        (0, 2, 3, 8),
        (5, 2, 3, 25),
        (3, 5, 3, 39),
        (7, 7, 3, 63),
        (8, 0, 4, 64),
        (0, 8, 4, 128),
        (15, 15, 4, 255),
    )
    for x, y, bits, expected_code in vectors:
        assert morton_encode(x, y, bits=bits) == expected_code
        actual_coordinates = morton_decompose(expected_code, bits=bits)
        expected_coordinates = (x, y)
        assert actual_coordinates == expected_coordinates


def test_morton_catalog_filters_candidates_with_exact_cartesian_oracle() -> None:
    catalog = MortonGeospatialCatalog(bits=3)
    strip = catalog.add("strip", 0, 0, 1, 4, label="vertical")
    hit = catalog.add("hit", 1, 0, 2, 1, label="point")
    candidates = catalog.approximate_candidates(1, 0, 2, 1)
    assert [record.handle for record in candidates] == [strip, hit]
    rows = [("strip", 0, 0, 1, 4), ("hit", 1, 0, 2, 1)]
    assert tuple(
        record.payload.item_id for record in catalog.search(1, 0, 2, 1)
    ) == search(rows, 1, 0, 2, 1)
    assert catalog.update(strip, min_x=2, max_x=3).handle == strip
    assert catalog.remove(hit).handle == hit
    assert catalog.snapshot().records[0].handle == strip


@pytest.mark.parametrize("bits", [True, "4", 0, 32])
def test_morton_width_has_explicit_integer_limits(bits: object) -> None:
    with pytest.raises(ValueError, match="integer from 1 through 31"):
        create_catalog(bits=bits)  # type: ignore[arg-type]


@pytest.mark.parametrize("coordinate", [True, 1.5])
def test_morton_encode_requires_integer_coordinates(coordinate: object) -> None:
    with pytest.raises(TypeError, match="coordinates must be integers"):
        morton_encode(coordinate, 0, bits=3)  # type: ignore[arg-type]


@pytest.mark.parametrize("coordinates", [(-1, 0), (0, 8)])
def test_morton_encode_rejects_coordinates_outside_width(
    coordinates: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="outside the configured Morton domain"):
        morton_encode(*coordinates, bits=3)


@pytest.mark.parametrize("code", [True, 1.5])
def test_morton_decompose_requires_an_integer_code(code: object) -> None:
    with pytest.raises(TypeError, match="code must be an integer"):
        morton_decompose(code, bits=3)  # type: ignore[arg-type]


@pytest.mark.parametrize("code", [-1, 64])
def test_morton_decompose_rejects_codes_outside_width(code: int) -> None:
    with pytest.raises(ValueError, match="outside the configured Morton domain"):
        morton_decompose(code, bits=3)


@pytest.mark.parametrize(
    ("coordinates", "error"),
    [
        ((False, 0, 1, 1), TypeError),
        ((0, 0, 0, 1), ValueError),
        ((0, -1, 1, 1), ValueError),
        ((0, 0, 9, 1), ValueError),
    ],
)
def test_morton_rectangles_are_typed_nonempty_and_domain_bounded(
    coordinates: tuple[object, object, object, object],
    error: type[Exception],
) -> None:
    catalog = create_catalog(bits=3)
    with pytest.raises(error, match="rectangle"):
        catalog.add(
            "region",
            *coordinates,  # type: ignore[arg-type]
            label="test",
        )


def test_morton_catalog_validates_labels_and_preserves_updated_identity() -> None:
    catalog = create_catalog(bits=4)
    with pytest.raises(ValueError, match="item_id must be a nonempty string"):
        catalog.add("", 0, 0, 1, 1, label="point")
    with pytest.raises(ValueError, match="label must be a nonempty string"):
        catalog.add("point", 0, 0, 1, 1, label="")

    handle = catalog.add("region", 1, 1, 3, 3, label="before")
    before = catalog.snapshot()
    unchanged = catalog.update(handle)
    assert unchanged.handle == handle
    updated = catalog.update(handle, min_x=2, min_y=3, max_x=5, max_y=6, label="after")
    assert updated.handle == handle
    assert updated.payload.min_x == 2
    assert updated.payload.min_y == 3
    assert updated.payload.max_x == 5
    assert updated.payload.max_y == 6
    assert updated.payload.label == "after"
    assert before.records[0].payload.label == "before"

    with pytest.raises(ValueError, match="rectangle must be nonempty"):
        catalog.update(handle, max_x=2)
    with pytest.raises(ValueError, match="label must be a nonempty string"):
        catalog.update(handle, label="")
    assert catalog.remove(handle).handle == handle
    with pytest.raises(KeyError):
        catalog.remove(handle)
