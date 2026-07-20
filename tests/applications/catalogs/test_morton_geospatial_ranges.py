"""Morton geospatial candidate/filter contract."""

from tests.oracles.applications.catalogs.morton_geospatial_ranges import search
from treemendous.applications.catalogs.morton_geospatial_ranges import (
    MortonGeospatialCatalog,
    morton_decompose,
    morton_encode,
)


def test_morton_catalog_filters_candidates_and_round_trips_integer_codes() -> None:
    for point in ((0, 0), (1, 2), (7, 7)):
        assert morton_decompose(morton_encode(*point, bits=3), bits=3) == point
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
