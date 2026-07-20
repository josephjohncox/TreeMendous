"""Timezone-normalized meeting-room example."""

from treemendous.applications.scheduling.meeting_rooms import (
    create_meeting_room_scheduler,
)


def main() -> None:
    scheduler = create_meeting_room_scheduler()
    booking = scheduler.book(
        "review", 12, 14, timezone_offset_slots=2, attendees=5,
        features=frozenset({"video"}), request_id="event-1",
    )
    print(booking.placement.resource, booking.utc_span.start, booking.utc_span.end)


if __name__ == "__main__":
    main()
