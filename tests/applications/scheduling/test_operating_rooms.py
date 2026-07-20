from __future__ import annotations

import pytest

from treemendous.applications.scheduling._common import SchedulingUnavailableError
from treemendous.applications.scheduling.operating_rooms import (
    ClinicalResource,
    OperatingRoom,
    OperatingRoomScheduler,
)


def test_procedure_room_staff_equipment_is_all_or_nothing() -> None:
    scheduler = OperatingRoomScheduler(
        (OperatingRoom("or", frozenset({"robotic"})),),
        (ClinicalResource("surgeon"),),
        (ClinicalResource("robot"),),
    )
    first = scheduler.schedule(
        "p1", 4, room_capabilities=frozenset({"robotic"}), staff=("surgeon",),
        equipment=("robot",), earliest_start=0, latest_end=4,
    )
    before = scheduler.snapshot()
    with pytest.raises(SchedulingUnavailableError) as raised:
        scheduler.schedule(
            "p2", 2, staff=("surgeon",), equipment=("robot",),
            earliest_start=1, latest_end=3,
        )
    assert raised.value.conflicts
    assert scheduler.snapshot() == before
    assert not scheduler.cancel("p1", first.id).reservation.active
