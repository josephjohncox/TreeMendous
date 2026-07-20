from time import perf_counter

import pytest

from tests.oracles.applications.scheduling.maintenance import valid_window
from tests.performance.applications.scheduling._shared import SmokeResult
from treemendous.applications.scheduling.maintenance import (
    MaintenanceScheduler,
    MaintenanceService,
)
from treemendous.domain import Span


def run_smoke(operations: int = 64) -> SmokeResult:
    scheduler = MaintenanceScheduler(
        (MaintenanceService("api", (Span(0, operations + 1),)),)
    )
    dependency: tuple[str, ...] = ()
    ready = 0
    started = perf_counter()
    for index in range(operations):
        task = f"t-{index}"
        booking = scheduler.schedule(
            task, "api", 1, dependencies=dependency, latest_end=operations + 1
        )
        start = booking.reservation.start
        is_valid = valid_window(
            start, start + 1, ((0, operations + 1),), (), ready
        )
        assert is_valid
        ready = start + 1
        dependency = (task,)
    return SmokeResult(operations, operations, perf_counter() - started)


@pytest.mark.benchmark
def test_maintenance_smoke_matches_oracle() -> None:
    assert run_smoke(16).oracle_checks == 16
