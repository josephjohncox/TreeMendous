from __future__ import annotations

import pytest

from treemendous.applications.scheduling._common import SchedulingUnavailableError
from treemendous.applications.scheduling.fleet_charging import (
    Charger,
    FleetChargingScheduler,
)


def test_charging_power_energy_dwell_and_bounded_policy() -> None:
    scheduler = FleetChargingScheduler(
        (
            Charger("slow", 5, frozenset({"ccs"})),
            Charger("fast", 10, frozenset({"ccs"})),
        ),
        max_session_slots=4,
    )
    session = scheduler.schedule(
        "van", 20, connector="ccs", arrival=5, departure=10, request_id="charge"
    )
    assert session.charger == "fast"
    assert session.duration == 2
    replay = scheduler.schedule(
        "van", 20, connector="ccs", arrival=5, departure=10, request_id="charge"
    )
    assert replay is session
    before = scheduler.snapshot()
    with pytest.raises(SchedulingUnavailableError):
        scheduler.schedule("truck", 100, connector="ccs", arrival=0, departure=4)
    assert scheduler.snapshot() == before
