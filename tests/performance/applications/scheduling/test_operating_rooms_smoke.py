from time import perf_counter

import pytest

from tests.oracles.applications.scheduling.operating_rooms import jointly_available
from tests.performance.applications.scheduling._shared import SmokeResult
from treemendous.applications.scheduling.operating_rooms import (
    ClinicalResource,
    OperatingRoom,
    OperatingRoomScheduler,
)


def run_smoke(operations: int = 64) -> SmokeResult:
    scheduler = OperatingRoomScheduler(
        (OperatingRoom("or"),), (ClinicalResource("doctor"),),
        (ClinicalResource("monitor"),),
    )
    started = perf_counter()
    for index in range(operations):
        booking = scheduler.schedule(
            f"p-{index}", 1, staff=("doctor",), equipment=("monitor",),
            earliest_start=index, latest_end=index + 1,
        )
        available = jointly_available(
            ("room:or", "staff:doctor", "equipment:monitor"), frozenset()
        )
        assert available
        assert booking.reservation.start == index
    return SmokeResult(operations, operations, perf_counter() - started)


@pytest.mark.benchmark
def test_operating_room_smoke_matches_oracle() -> None:
    assert run_smoke(16).oracle_checks == 16
