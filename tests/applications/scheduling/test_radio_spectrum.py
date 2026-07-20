from __future__ import annotations

import pytest

from treemendous.applications.scheduling.radio_spectrum import (
    RadioSpectrumScheduler,
    SpectrumConflictError,
)


def test_spectrum_uses_exact_channel_time_rectangles_and_guard_bands() -> None:
    scheduler = RadioSpectrumScheduler(20)
    first = scheduler.reserve(
        "tx1", 5, 2, 10, 20, guard_channels=1, request_id="packet"
    )
    touching_time = scheduler.reserve("tx2", 5, 2, 20, 25, guard_channels=1)
    with pytest.raises(SpectrumConflictError) as raised:
        scheduler.reserve("tx3", 8, 1, 15, 16, guard_channels=1)
    assert len(raised.value.conflict.conflicting_ids) == 1
    assert raised.value.conflict.conflicting_ids[0] == first.id
    replay = scheduler.reserve(
        "tx1", 5, 2, 10, 20, guard_channels=1, request_id="packet"
    )
    assert replay is first
    assert len(scheduler.snapshot().geometry.entries) == 2
    scheduler.cancel("tx1", first.id)
    assert len(scheduler.snapshot().geometry.entries) == 1
    assert touching_time.active
