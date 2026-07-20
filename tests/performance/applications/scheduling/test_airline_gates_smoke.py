from time import perf_counter

import pytest

from tests.oracles.applications.scheduling.airline_gates import compatible, occupied
from tests.performance.applications.scheduling._shared import SmokeResult
from treemendous.applications.scheduling.airline_gates import AirlineGateScheduler, Gate


def run_smoke(operations: int = 64) -> SmokeResult:
    gate = Gate("gate", frozenset({"A320"}))
    scheduler = AirlineGateScheduler((gate,))
    started = perf_counter()
    for index in range(operations):
        start = index * 3 + 1
        placement = scheduler.assign(
            f"f-{index}", start, start + 1, aircraft_type="A320",
            turnaround_before=1, turnaround_after=1,
        )
        supports_aircraft = compatible(gate.aircraft_types, "A320")
        expected_start, expected_end = occupied(start, start + 1, 1, 1)
        assert supports_aircraft
        assert placement.reservation.occupied_span.start == expected_start
        assert placement.reservation.occupied_span.end == expected_end
    return SmokeResult(operations, operations, perf_counter() - started)


@pytest.mark.benchmark
def test_airline_gate_smoke_matches_oracle() -> None:
    assert run_smoke(16).oracle_checks == 16
