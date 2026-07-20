"""Use Morton bands as candidates and exact Cartesian filtering."""

from treemendous.applications.catalogs.morton_geospatial_ranges import (
    MortonGeospatialCatalog,
)


def main() -> None:
    catalog = MortonGeospatialCatalog(bits=8)
    catalog.add("park", 10, 10, 30, 25, label="city park")
    print([record.payload.label for record in catalog.search(20, 20, 40, 40)])


if __name__ == "__main__":
    main()
