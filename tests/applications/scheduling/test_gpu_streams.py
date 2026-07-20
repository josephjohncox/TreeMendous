from __future__ import annotations

import pytest

from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications._shared.reservations import ReservationStatus
from treemendous.applications.scheduling.gpu_streams import (
    GPUDevice,
    GPUStream,
    GPUStreamScheduler,
)


def test_gpu_stream_compatibility_dependency_ready_and_capacity() -> None:
    scheduler = GPUStreamScheduler(
        (
            GPUDevice(
                "a",
                CapacityVector(memory=8, slots=1),
                frozenset({"compute"}),
                (GPUStream("s0", frozenset({"compute"})),),
            ),
        )
    )
    first = scheduler.schedule(
        "k1",
        3,
        CapacityVector(memory=4, slots=1),
        compatibility="compute",
        dependency_ready_times={"copy": 5},
        latest_end=10,
        request_id="once",
    )
    assert first.device == "a"
    assert first.stream == "s0"
    assert first.start == 5
    second = scheduler.schedule(
        "k2",
        2,
        CapacityVector(memory=4, slots=1),
        compatibility="compute",
        earliest_start=5,
        latest_end=10,
    )
    assert second.start == 8
    replay = scheduler.schedule(
        "k1",
        3,
        CapacityVector(memory=4, slots=1),
        compatibility="compute",
        dependency_ready_times={"copy": 5},
        latest_end=10,
        request_id="once",
    )
    assert replay is first


def test_gpu_resource_identities_do_not_collide_and_cancel_round_trips() -> None:
    scheduler = GPUStreamScheduler(
        (
            GPUDevice(
                "a",
                CapacityVector(slots=1),
                frozenset({"compute"}),
                (GPUStream("b:c", frozenset({"compute"})),),
            ),
            GPUDevice(
                "a:b",
                CapacityVector(slots=1),
                frozenset({"compute"}),
                (GPUStream("c", frozenset({"compute"})),),
            ),
        )
    )

    first = scheduler.schedule(
        "k1",
        1,
        CapacityVector(slots=1),
        compatibility="compute",
        latest_end=1,
    )
    second = scheduler.schedule(
        "k2",
        1,
        CapacityVector(slots=1),
        compatibility="compute",
        latest_end=1,
    )

    first_resources = {
        requirement.resource for requirement in first.reservation.requirements
    }
    second_resources = {
        requirement.resource for requirement in second.reservation.requirements
    }
    assert first_resources.isdisjoint(second_resources)

    cancelled = scheduler.cancel("k2", second.id)
    assert cancelled.device == "a:b"
    assert cancelled.stream == "c"
    assert cancelled.reservation.status is ReservationStatus.CANCELLED
    reservations = {
        reservation.id: reservation for reservation in scheduler.snapshot().reservations
    }
    assert reservations[second.id] == cancelled.reservation


def test_gpu_changed_idempotent_request_is_failure_atomic() -> None:
    scheduler = GPUStreamScheduler(
        (
            GPUDevice(
                "gpu",
                CapacityVector(slots=1),
                frozenset({"compute"}),
                (GPUStream("stream", frozenset({"compute"})),),
            ),
        )
    )
    scheduler.schedule(
        "kernel",
        1,
        CapacityVector(slots=1),
        compatibility="compute",
        latest_end=2,
        request_id="request",
    )
    before = scheduler.snapshot()

    with pytest.raises(ValueError, match="idempotency key"):
        scheduler.schedule(
            "kernel",
            2,
            CapacityVector(slots=1),
            compatibility="compute",
            latest_end=2,
            request_id="request",
        )

    assert scheduler.snapshot() == before
