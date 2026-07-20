"""Validated benchmark for Morton candidates with exact spatial filtering."""

from __future__ import annotations

from random import Random
from typing import Any

from tests.oracles.applications.catalogs.morton_geospatial_ranges import search
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from treemendous.applications.catalogs.morton_geospatial_ranges import (
    GeoRecord,
    MortonGeospatialCatalog,
)

_MAX_OPERATIONS = 10_000


def _parameters(operations: int, seed: int) -> Random:
    if isinstance(operations, bool) or not isinstance(operations, int):
        raise TypeError("operations must be an integer")
    if not 1 <= operations <= _MAX_OPERATIONS:
        raise ValueError(f"operations must be between 1 and {_MAX_OPERATIONS}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return Random(seed)


def _record(record: GeoRecord) -> tuple[Any, ...]:
    region = record.payload
    return (
        record.handle.owner,
        record.handle.sequence,
        "engine-lineage",
        record.start,
        record.end,
        record.insertion_order,
        region.item_id,
        region.min_x,
        region.min_y,
        region.max_x,
        region.max_y,
        region.label,
    )


def _state(catalog: MortonGeospatialCatalog) -> dict[str, Any]:
    snapshot = catalog.snapshot()
    return {
        "bits": catalog.bits,
        "records": tuple(_record(record) for record in snapshot.records),
        "next_sequences": snapshot.next_sequences,
        "next_insertion_order": snapshot.next_insertion_order,
    }


def run_benchmark(operations: int = 500, seed: int = 0) -> ApplicationSample:
    """Run bounded searches and validate exact rectangles by Cartesian scan."""
    random = _parameters(operations, seed)
    catalog = MortonGeospatialCatalog(bits=10)
    rows: list[tuple[str, int, int, int, int]] = []
    for index in range(200):
        x = index % 50 * 10
        y = index // 50 * 20
        item_id = f"g{index}"
        catalog.add(item_id, x, y, x + 8, y + 8, label="region")
        rows.append((item_id, x, y, x + 8, y + 8))

    commands = tuple(random.randrange(500) for _ in range(operations))
    expected_state = _state(catalog)
    by_id = {row[6]: row for row in expected_state["records"]}

    def execute() -> tuple[tuple[GeoRecord, ...], ...]:
        return tuple(catalog.search(x, 0, x + 30, 100) for x in commands)

    def observe(raw: tuple[tuple[GeoRecord, ...], ...]) -> ApplicationOutcome:
        results = tuple(tuple(_record(record) for record in result) for result in raw)
        return ApplicationOutcome(
            results,
            _state(catalog),
            {
                "query_calls": operations,
                "returned_regions": sum(len(result) for result in results),
            },
        )

    def oracle() -> ApplicationOutcome:
        ids = tuple(search(rows, x, 0, x + 30, 100) for x in commands)
        results = tuple(tuple(by_id[item_id] for item_id in result) for result in ids)
        return ApplicationOutcome(
            results,
            expected_state,
            {
                "query_calls": operations,
                "returned_regions": sum(len(result) for result in results),
            },
        )

    return run_application_case(
        scenario_id="catalog-morton-geospatial-search",
        operations=operations,
        execute=execute,
        observe=observe,
        oracle=oracle,
    )


def run_smoke(iterations: int = 500, seed: int = 0) -> ApplicationSample:
    """Delegate the legacy smoke entry point to the validated benchmark."""
    return run_benchmark(operations=iterations, seed=seed)


if __name__ == "__main__":
    print(run_smoke().to_dict())
