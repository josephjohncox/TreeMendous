from __future__ import annotations

import pytest

from treemendous.applications.scheduling._common import SchedulingUnavailableError
from treemendous.applications.scheduling.meeting_rooms import (
    MeetingRoom,
    MeetingRoomScheduler,
)


def test_meetings_normalize_slots_and_enforce_attendees_and_features() -> None:
    scheduler = MeetingRoomScheduler(
        (
            MeetingRoom("small", 4, frozenset({"whiteboard"})),
            MeetingRoom("video", 10, frozenset({"video"}), timezone_offset_slots=2),
        )
    )
    booking = scheduler.book(
        "m1", 10, 12, timezone_offset_slots=2, attendees=8,
        features=frozenset({"video"}), request_id="calendar-event",
    )
    assert booking.placement.resource == "video"
    assert booking.utc_span.start == 8
    before = scheduler.snapshot()
    with pytest.raises(SchedulingUnavailableError):
        scheduler.book(
            "m2", 8, 10, timezone_offset_slots=0, attendees=8,
            features=frozenset({"video"}),
        )
    assert scheduler.snapshot() == before
    scheduler.cancel("m1", booking.placement.id)
