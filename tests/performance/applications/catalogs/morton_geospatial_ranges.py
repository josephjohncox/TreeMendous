"""Smoke benchmark for Morton candidates plus exact filtering."""

from time import perf_counter

from treemendous.applications.catalogs.morton_geospatial_ranges import (
    MortonGeospatialCatalog,
)


def run_smoke(iterations: int = 500) -> float:
    catalog = MortonGeospatialCatalog(bits=10)
    for index in range(200):
        x = index % 50 * 10
        y = index // 50 * 20
        catalog.add(f"g{index}", x, y, x + 8, y + 8, label="region")
    started = perf_counter()
    for index in range(iterations):
        x = index % 500
        catalog.search(x, 0, x + 30, 100)
    return perf_counter() - started


if __name__ == "__main__":
    print(run_smoke())
