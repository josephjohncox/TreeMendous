from time import perf_counter

import pytest

from tests.oracles.applications.scheduling.radio_spectrum import overlaps
from tests.performance.applications.scheduling._shared import SmokeResult
from treemendous.applications.scheduling.radio_spectrum import RadioSpectrumScheduler


def run_smoke(operations: int = 64) -> SmokeResult:
    scheduler = RadioSpectrumScheduler(16)
    previous: tuple[int, int, int, int] | None = None
    started = perf_counter()
    for index in range(operations):
        reservation = scheduler.reserve("tx", 4, 2, index, index + 1)
        current = (
            reservation.channel_start, reservation.channel_end,
            reservation.start, reservation.end,
        )
        if previous is not None:
            assert not overlaps(previous, current)
        previous = current
    return SmokeResult(operations, operations, perf_counter() - started)


@pytest.mark.benchmark
def test_radio_spectrum_smoke_matches_oracle() -> None:
    assert run_smoke(16).oracle_checks == 16
