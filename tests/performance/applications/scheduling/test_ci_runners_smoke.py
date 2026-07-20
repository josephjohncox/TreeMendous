from time import perf_counter

import pytest

from tests.oracles.applications.scheduling.ci_runners import expected_runner
from tests.performance.applications.scheduling._shared import SmokeResult
from treemendous.applications.scheduling.ci_runners import CIRunnerScheduler, Runner


def run_smoke(operations: int = 64) -> SmokeResult:
    runners = (Runner("runner", frozenset({"linux"}), 2),)
    scheduler = CIRunnerScheduler(runners)
    reference = expected_runner((("runner", frozenset({"linux"}), 2),), frozenset({"linux"}))
    started = perf_counter()
    for index in range(operations):
        placement = scheduler.schedule(
            f"ci-{index}", 1, labels=frozenset({"linux"}),
            release_time=index, deadline=index + 1,
        )
        assert placement.resource == reference
    return SmokeResult(operations, operations, perf_counter() - started)


@pytest.mark.benchmark
def test_ci_runner_smoke_matches_oracle() -> None:
    assert run_smoke(16).oracle_checks == 16
