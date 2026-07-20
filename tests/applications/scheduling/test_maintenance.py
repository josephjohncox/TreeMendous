from __future__ import annotations

import pytest

from treemendous.applications.scheduling._common import SchedulingUnavailableError
from treemendous.applications.scheduling.maintenance import (
    MaintenanceScheduler,
    MaintenanceService,
)
from treemendous.domain import Span


def test_maintenance_dependencies_blackouts_and_windows() -> None:
    scheduler = MaintenanceScheduler(
        (MaintenanceService("api", (Span(0, 20),), (Span(5, 8),)),)
    )
    first = scheduler.schedule("db", "api", 3, latest_end=20)
    second = scheduler.schedule(
        "app",
        "api",
        3,
        dependencies=("db",),
        earliest_start=0,
        latest_end=20,
        request_id="deploy",
    )
    assert first.reservation.start == 0
    assert second.reservation.start == 8
    replay = scheduler.schedule(
        "app",
        "api",
        3,
        dependencies=("db",),
        earliest_start=0,
        latest_end=20,
        request_id="deploy",
    )
    assert replay is second
    with pytest.raises(KeyError, match="unknown dependency"):
        scheduler.schedule("bad", "api", 1, dependencies=("missing",), latest_end=20)
    scheduler.cancel("db")
    with pytest.raises(ValueError, match="cancelled"):
        scheduler.schedule("later", "api", 1, dependencies=("db",), latest_end=20)
    with pytest.raises(SchedulingUnavailableError):
        scheduler.schedule("outside", "api", 2, earliest_start=19, latest_end=21)
