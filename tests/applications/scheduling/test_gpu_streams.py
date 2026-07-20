from __future__ import annotations

from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications.scheduling.gpu_streams import (
    GPUDevice,
    GPUStream,
    GPUStreamScheduler,
)


def test_gpu_stream_compatibility_dependency_ready_and_capacity() -> None:
    scheduler = GPUStreamScheduler(
        (
            GPUDevice(
                "a", CapacityVector(memory=8, slots=1), frozenset({"compute"}),
                (GPUStream("s0", frozenset({"compute"})),),
            ),
        )
    )
    first = scheduler.schedule(
        "k1", 3, CapacityVector(memory=4, slots=1), compatibility="compute",
        dependency_ready_times={"copy": 5}, latest_end=10, request_id="once",
    )
    assert first.device == "a"
    assert first.stream == "s0"
    assert first.start == 5
    second = scheduler.schedule(
        "k2", 2, CapacityVector(memory=4, slots=1), compatibility="compute",
        earliest_start=5, latest_end=10,
    )
    assert second.start == 8
    replay = scheduler.schedule(
        "k1", 3, CapacityVector(memory=4, slots=1), compatibility="compute",
        dependency_ready_times={"copy": 5}, latest_end=10, request_id="once",
    )
    assert replay is first
