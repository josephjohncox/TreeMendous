from __future__ import annotations

import pytest

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
        "film", first.id, duration=2, earliest_start=2, latest_end=6,
        request_id="retry-1",
    )
    assert retry.frames == first.frames
    assert retry.attempt == 2
    history = scheduler.snapshot().assignments
    assert history[0].status is ChunkStatus.RETRIED
    assert history[1].status is ChunkStatus.ACTIVE
