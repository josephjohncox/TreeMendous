"""Atomic operating-room resource booking example."""

from treemendous.applications.scheduling.operating_rooms import (
    create_operating_room_scheduler,
)


def main() -> None:
    scheduler = create_operating_room_scheduler()
    booking = scheduler.schedule(
        "procedure", 3, room_capabilities=frozenset({"general"}),
        staff=("surgeon-1",), equipment=("monitor-1",),
        earliest_start=4, latest_end=10, request_id="case-1",
    )
    print(booking.room, booking.reservation.start, booking.reservation.end)


if __name__ == "__main__":
    main()
