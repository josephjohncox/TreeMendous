from time import perf_counter

import pytest

from tests.oracles.applications.scheduling.render_farm import chunks_overlap
from tests.performance.applications.scheduling._shared import SmokeResult
from treemendous.applications.scheduling.render_farm import (
    RenderFarmScheduler,
    RenderWorker,
)


def run_smoke(operations: int = 64) -> SmokeResult:
    scheduler = RenderFarmScheduler((RenderWorker("worker"),))
    previous: tuple[int, int] | None = None
    started = perf_counter()
    for index in range(operations):
        assignment = scheduler.assign_chunk(
            "film", index * 10, 10, 1, earliest_start=index, latest_end=index + 1
        )
        current = (assignment.frames.start, assignment.frames.end)
        if previous is not None:
            assert not chunks_overlap(previous, current)
        previous = current
    return SmokeResult(operations, operations, perf_counter() - started)


@pytest.mark.benchmark
def test_render_farm_smoke_matches_oracle() -> None:
    assert run_smoke(16).oracle_checks == 16
