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
        "m1",
        10,
        12,
        timezone_offset_slots=2,
        attendees=8,
        features=frozenset({"video"}),
        request_id="calendar-event",
    )
    assert booking.placement.resource == "video"
    assert booking.utc_span.start == 8
    before = scheduler.snapshot()
    with pytest.raises(SchedulingUnavailableError):
        scheduler.book(
            "m2",
            8,
            10,
            timezone_offset_slots=0,
            attendees=8,
            features=frozenset({"video"}),
        )
    assert scheduler.snapshot() == before
    scheduler.cancel("m1", booking.placement.id)


def test_meeting_booking_exact_replay_returns_original_booking() -> None:
    scheduler = MeetingRoomScheduler((MeetingRoom("room", 8, frozenset({"video"})),))
    booking = scheduler.book(
        "meeting",
        10,
        12,
        timezone_offset_slots=2,
        attendees=3,
        features=frozenset({"video"}),
        request_id="request",
    )

    replay = scheduler.book(
        "meeting",
        10,
        12,
        timezone_offset_slots=2,
        attendees=3,
        features=frozenset({"video"}),
        request_id="request",
    )

    assert replay is booking
    assert scheduler.snapshot().reservations == (booking.placement.reservation,)


def test_meeting_idempotency_fingerprints_attendees_failure_atomically() -> None:
    scheduler = MeetingRoomScheduler((MeetingRoom("room", 8, frozenset({"video"})),))
    scheduler.book(
        "meeting",
        10,
        12,
        timezone_offset_slots=2,
        attendees=3,
        features=frozenset({"video"}),
        request_id="request",
    )
    before = scheduler.snapshot()

    with pytest.raises(ValueError, match="idempotency key"):
        scheduler.book(
            "meeting",
            10,
            12,
            timezone_offset_slots=2,
            attendees=4,
            features=frozenset({"video"}),
            request_id="request",
        )

    assert scheduler.snapshot() == before


def test_meeting_idempotency_fingerprints_local_timezone_inputs() -> None:
    scheduler = MeetingRoomScheduler((MeetingRoom("room", 8, frozenset({"video"})),))
    booking = scheduler.book(
        "meeting",
        10,
        12,
        timezone_offset_slots=2,
        attendees=3,
        features=frozenset({"video"}),
        request_id="request",
    )
    before = scheduler.snapshot()

    # This describes the same UTC span, but it is not the same public request.
    with pytest.raises(ValueError, match="idempotency key"):
        scheduler.book(
            "meeting",
            11,
            13,
            timezone_offset_slots=3,
            attendees=3,
            features=frozenset({"video"}),
            request_id="request",
        )

    assert scheduler.snapshot() == before
    replay = scheduler.book(
        "meeting",
        10,
        12,
        timezone_offset_slots=2,
        attendees=3,
        features=frozenset({"video"}),
        request_id="request",
    )
    assert replay is booking
