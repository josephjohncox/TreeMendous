from __future__ import annotations

import pytest

from treemendous.applications._shared.reservations import ReservationStatus
from treemendous.applications.scheduling._common import SchedulingUnavailableError
from treemendous.applications.scheduling.render_farm import (
    ChunkStatus,
    RenderFarmScheduler,
    RenderWorker,
)


def test_render_chunks_do_not_overlap_and_retry_preserves_history() -> None:
    scheduler = RenderFarmScheduler((RenderWorker("worker"),))
    first = scheduler.assign_chunk(
        "film", 0, 10, 2, earliest_start=0, latest_end=4, request_id="a1"
    )
    with pytest.raises(ValueError, match="overlaps"):
        scheduler.assign_chunk("film", 5, 10, 2, earliest_start=2, latest_end=6)
    retry = scheduler.retry(
        "film",
        first.id,
        duration=2,
        earliest_start=2,
        latest_end=6,
        request_id="retry-1",
    )
    assert retry.frames == first.frames
    assert retry.attempt == 2
    snapshot = scheduler.snapshot()
    history = {assignment.id: assignment for assignment in snapshot.assignments}
    assert history[first.id].status is ChunkStatus.RETRIED
    assert history[first.id].placement.reservation.status is ReservationStatus.CANCELLED
    assert history[retry.id].status is ChunkStatus.ACTIVE
    assert history[retry.id].placement.reservation.status is ReservationStatus.ACTIVE
    reservations = {
        reservation.id: reservation
        for reservation in snapshot.reservations.reservations
    }
    assert history[first.id].placement.reservation == reservations[first.id]
    assert history[retry.id].placement.reservation == reservations[retry.id]


def test_render_cancel_uses_cancelled_placement_in_history() -> None:
    scheduler = RenderFarmScheduler((RenderWorker("worker"),))
    assignment = scheduler.assign_chunk(
        "film", 0, 10, 2, earliest_start=0, latest_end=4, request_id="a1"
    )

    cancelled = scheduler.cancel("film", assignment.id)

    assert cancelled.status is ChunkStatus.CANCELLED
    assert cancelled.placement.reservation.status is ReservationStatus.CANCELLED
    snapshot = scheduler.snapshot()
    assert snapshot.assignments == (cancelled,)
    assert snapshot.reservations.reservations == (cancelled.placement.reservation,)


def test_render_retry_rejects_assignment_request_id_without_mutation() -> None:
    scheduler = RenderFarmScheduler((RenderWorker("worker"),))
    assignment = scheduler.assign_chunk(
        "film", 0, 10, 2, earliest_start=0, latest_end=4, request_id="a1"
    )
    before = scheduler.snapshot()

    with pytest.raises(ValueError, match="replacement assignment"):
        scheduler.retry(
            "film",
            assignment.id,
            duration=2,
            earliest_start=0,
            latest_end=4,
            request_id="a1",
        )

    assert scheduler.snapshot() == before


def test_render_retry_is_idempotent_for_the_source_attempt() -> None:
    scheduler = RenderFarmScheduler((RenderWorker("worker"),))
    assignment = scheduler.assign_chunk(
        "film", 0, 10, 2, earliest_start=0, latest_end=4, request_id="a1"
    )
    replacement = scheduler.retry(
        "film",
        assignment.id,
        duration=2,
        earliest_start=2,
        latest_end=6,
        request_id="retry-1",
    )
    after = scheduler.snapshot()

    replay = scheduler.retry(
        "film",
        assignment.id,
        duration=2,
        earliest_start=2,
        latest_end=6,
        request_id="retry-1",
    )

    assert replay is replacement
    assert scheduler.snapshot() == after

    with pytest.raises(ValueError, match="different retry"):
        scheduler.retry(
            "film",
            replacement.id,
            duration=2,
            earliest_start=4,
            latest_end=8,
            request_id="retry-1",
        )
    assert scheduler.snapshot() == after


def test_failed_render_retry_is_atomic() -> None:
    scheduler = RenderFarmScheduler((RenderWorker("worker"),))
    assignment = scheduler.assign_chunk(
        "film", 0, 10, 2, earliest_start=0, latest_end=2, request_id="a1"
    )
    before = scheduler.snapshot()

    with pytest.raises(SchedulingUnavailableError):
        scheduler.retry(
            "film",
            assignment.id,
            duration=2,
            earliest_start=0,
            latest_end=2,
            request_id="retry-1",
        )

    assert scheduler.snapshot() == before
