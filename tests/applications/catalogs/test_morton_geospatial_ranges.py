"""Morton geospatial candidate/filter contract."""

from tests.oracles.applications.catalogs.morton_geospatial_ranges import search
from treemendous.applications.catalogs.morton_geospatial_ranges import (
    MortonGeospatialCatalog,
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
        assert morton_decompose(expected_code, bits=bits) == (x, y)


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
