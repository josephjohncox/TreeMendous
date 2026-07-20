from time import perf_counter

import pytest

from tests.oracles.applications.scheduling.gpu_streams import expected_pair
from tests.performance.applications.scheduling._shared import SmokeResult
from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications.scheduling.gpu_streams import (
    GPUDevice,
    GPUStream,
    GPUStreamScheduler,
)


def run_smoke(operations: int = 64) -> SmokeResult:
    stream = GPUStream("s0", frozenset({"compute"}))
    device = GPUDevice("gpu", CapacityVector(memory=8, slots=1), frozenset({"compute"}), (stream,))
    scheduler = GPUStreamScheduler((device,))
    reference = expected_pair((("gpu", device.compatibility, (("s0", stream.compatibility),)),), "compute")
    started = perf_counter()
    for index in range(operations):
        placement = scheduler.schedule(
            f"k-{index}", 1, CapacityVector(memory=1, slots=1),
            compatibility="compute", dependency_ready_times={"ready": index},
            latest_end=index + 1,
        )
        assert reference is not None
        assert placement.device == reference[0]
        assert placement.stream == reference[1]
    return SmokeResult(operations, operations, perf_counter() - started)


@pytest.mark.benchmark
def test_gpu_stream_smoke_matches_oracle() -> None:
    assert run_smoke(16).oracle_checks == 16
