from time import perf_counter

import pytest

from tests.oracles.applications.scheduling.lab_instruments import calibrated
from tests.performance.applications.scheduling._shared import SmokeResult
from treemendous.applications.scheduling.lab_instruments import (
    LabInstrument,
    LabInstrumentScheduler,
)
from treemendous.domain import Span


def run_smoke(operations: int = 64) -> SmokeResult:
    window = Span(0, operations * 2 + 1)
    scheduler = LabInstrumentScheduler(
        (LabInstrument("scope", frozenset({"image"}), (window,), cleanup_slots=1),)
    )
    started = perf_counter()
    for index in range(operations):
        placement = scheduler.book(
            f"e-{index}", 1, capabilities=frozenset({"image"}),
            earliest_start=index * 2, latest_end=index * 2 + 1,
        )
        is_calibrated = calibrated(
            placement.start,
            placement.end,
            1,
            ((window.start, window.end),),
        )
        assert is_calibrated
    return SmokeResult(operations, operations, perf_counter() - started)


@pytest.mark.benchmark
def test_lab_instrument_smoke_matches_oracle() -> None:
    assert run_smoke(16).oracle_checks == 16
