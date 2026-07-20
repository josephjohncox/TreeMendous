from time import perf_counter

import pytest

from tests.oracles.applications.scheduling.meeting_rooms import expected_room, normalize
from tests.performance.applications.scheduling._shared import SmokeResult
from treemendous.applications.scheduling.meeting_rooms import (
    MeetingRoom,
    MeetingRoomScheduler,
)


def run_smoke(operations: int = 64) -> SmokeResult:
    rooms = (MeetingRoom("room", 8, frozenset({"video"})),)
    scheduler = MeetingRoomScheduler(rooms)
    reference = expected_room((("room", 8, frozenset({"video"})),), 4, frozenset({"video"}))
    started = perf_counter()
    for index in range(operations):
        booking = scheduler.book(
            f"m-{index}", index + 2, index + 3, timezone_offset_slots=2,
            attendees=4, features=frozenset({"video"}),
        )
        expected_start, expected_end = normalize(index + 2, index + 3, 2)
        assert booking.placement.resource == reference
        assert booking.utc_span.start == expected_start
        assert booking.utc_span.end == expected_end
    return SmokeResult(operations, operations, perf_counter() - started)


@pytest.mark.benchmark
def test_meeting_room_smoke_matches_oracle() -> None:
    assert run_smoke(16).oracle_checks == 16
