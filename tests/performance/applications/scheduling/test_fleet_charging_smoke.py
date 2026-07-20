from time import perf_counter

import pytest

from tests.oracles.applications.scheduling.fleet_charging import duration, feasible
from tests.performance.applications.scheduling._shared import SmokeResult
from treemendous.applications.scheduling.fleet_charging import (
    Charger,
    FleetChargingScheduler,
)


def run_smoke(operations: int = 64) -> SmokeResult:
    scheduler = FleetChargingScheduler(
        (Charger("charger", 10, frozenset({"ccs"})),), max_session_slots=4
    )
    started = perf_counter()
    for index in range(operations):
        session = scheduler.schedule(
            f"v-{index}", 10, connector="ccs", arrival=index, departure=index + 1
        )
        assert feasible(10, 10, 1, 4)
        assert session.duration == duration(10, 10)
    return SmokeResult(operations, operations, perf_counter() - started)


@pytest.mark.benchmark
def test_fleet_charging_smoke_matches_oracle() -> None:
    assert run_smoke(16).oracle_checks == 16
