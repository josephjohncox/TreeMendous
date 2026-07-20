from time import perf_counter

import pytest

from tests.oracles.applications.scheduling.warehouse_docks import compatible, occupied
from tests.performance.applications.scheduling._shared import SmokeResult
from treemendous.applications.scheduling.warehouse_docks import (
    Dock,
    WarehouseDockScheduler,
)


def run_smoke(operations: int = 64) -> SmokeResult:
    dock = Dock("dock", frozenset({"dry"}))
    scheduler = WarehouseDockScheduler((dock,))
    started = perf_counter()
    for index in range(operations):
        start = index * 3 + 1
        placement = scheduler.book(
            f"c-{index}", 1, cargo_type="dry", earliest_start=start,
            latest_end=start + 1, handling_before=1, handling_after=1,
        )
        expected_start, expected_end = occupied(start, start + 1, 1, 1)
        assert compatible(dock.cargo_types, "dry")
        assert placement.reservation.occupied_span.start == expected_start
        assert placement.reservation.occupied_span.end == expected_end
    return SmokeResult(operations, operations, perf_counter() - started)


@pytest.mark.benchmark
def test_warehouse_dock_smoke_matches_oracle() -> None:
    assert run_smoke(16).oracle_checks == 16
