from __future__ import annotations

import pytest

from treemendous.applications.scheduling._common import SchedulingUnavailableError
from treemendous.applications.scheduling.warehouse_docks import (
    Dock,
    WarehouseDockScheduler,
)


def test_dock_cargo_compatibility_handling_buffers_and_cancel() -> None:
    scheduler = WarehouseDockScheduler((Dock("cold", frozenset({"frozen"})),))
    first = scheduler.book(
        "carrier-a",
        2,
        cargo_type="frozen",
        earliest_start=10,
        latest_end=12,
        handling_before=1,
        handling_after=2,
    )
    with pytest.raises(SchedulingUnavailableError):
        scheduler.book(
            "carrier-b", 1, cargo_type="frozen", earliest_start=13, latest_end=14
        )
    scheduler.cancel("carrier-a", first.id)
    second = scheduler.book(
        "carrier-b", 1, cargo_type="frozen", earliest_start=13, latest_end=14
    )
    assert second.start == 13
