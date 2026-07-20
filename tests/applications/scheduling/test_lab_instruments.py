from __future__ import annotations

import pytest

from treemendous.applications.scheduling._common import SchedulingUnavailableError
from treemendous.applications.scheduling.lab_instruments import (
    LabInstrument,
    LabInstrumentScheduler,
)
from treemendous.domain import Span


def test_lab_capability_calibration_and_cleanup() -> None:
    scheduler = LabInstrumentScheduler(
        (LabInstrument("scope", frozenset({"imaging"}), (Span(10, 20),), 2),)
    )
    first = scheduler.book(
        "e1", 3, capabilities=frozenset({"imaging"}), earliest_start=10,
        latest_end=20,
    )
    second = scheduler.book(
        "e2", 3, capabilities=frozenset({"imaging"}), earliest_start=10,
        latest_end=20,
    )
    assert first.start == 10
    assert second.start == 15
    with pytest.raises(SchedulingUnavailableError):
        scheduler.book(
            "e3", 1, capabilities=frozenset({"mass-spec"}),
            earliest_start=10, latest_end=20,
        )
