from __future__ import annotations

import pytest

from treemendous.applications.scheduling._common import SchedulingUnavailableError
from treemendous.applications.scheduling.airline_gates import (
    AirlineGateScheduler,
    Gate,
)


def test_gate_aircraft_compatibility_and_turnaround_buffers() -> None:
    scheduler = AirlineGateScheduler((Gate("g1", frozenset({"A320"})),))
    first = scheduler.assign(
        "f1", 10, 20, aircraft_type="A320", turnaround_before=2,
        turnaround_after=2,
    )
    with pytest.raises(SchedulingUnavailableError):
        scheduler.assign("f2", 21, 23, aircraft_type="A320")
    touching = scheduler.assign("f3", 22, 24, aircraft_type="A320")
    assert first.reservation.occupied_span.end == touching.start
