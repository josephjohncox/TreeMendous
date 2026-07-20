"""Meeting-room booking over timezone-normalized integer slots."""

from __future__ import annotations

from dataclasses import dataclass

from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications._shared.reservations import ReservationSnapshot
from treemendous.applications.scheduling._common import Placement, integer, names, text
from treemendous.applications.scheduling._placement import (
    BoundedPlacementEngine,
    LabeledResource,
)
from treemendous.domain import Span, validate_coordinate


@dataclass(frozen=True)
class MeetingRoom:
    """A room with attendee capacity, features, and informational UTC offset."""

    name: str
    attendee_capacity: int
    features: frozenset[str] = frozenset()
    timezone_offset_slots: int = 0

    def __post_init__(self) -> None:
        text(self.name, "room name")
        integer(self.attendee_capacity, "attendee_capacity", minimum=1)
        names(self.features, "features")
        validate_coordinate(self.timezone_offset_slots, "timezone_offset_slots")


@dataclass(frozen=True)
class MeetingBooking:
    """A booking whose span is normalized to UTC integer slots."""

    placement: Placement
    attendees: int
    utc_span: Span


class MeetingRoomScheduler:
    """Books one compatible room; recurring calendars and DST rules are out of scope."""

    def __init__(self, rooms: tuple[MeetingRoom, ...]) -> None:
        if not rooms:
            raise ValueError("at least one room is required")
        self._rooms = {room.name: room for room in rooms}
        if len(self._rooms) != len(rooms):
            raise ValueError("room names must be unique")
        self._engine = BoundedPlacementEngine(
            tuple(
                LabeledResource(room.name, CapacityVector(units=1), room.features)
                for room in rooms
            )
        )

    def book(
        self,
        meeting_id: str,
        local_start: int,
        local_end: int,
        *,
        timezone_offset_slots: int,
        attendees: int,
        features: frozenset[str] = frozenset(),
        request_id: str | None = None,
    ) -> MeetingBooking:
        validate_coordinate(timezone_offset_slots, "timezone_offset_slots")
        local = Span(local_start, local_end)
        integer(attendees, "attendees", minimum=1)
        names(features, "features")
        utc = Span(
            local.start - timezone_offset_slots,
            local.end - timezone_offset_slots,
        )
        eligible = frozenset(
            room.name
            for room in self._rooms.values()
            if attendees <= room.attendee_capacity
        )
        placement = self._engine.place(
            meeting_id,
            utc.length,
            CapacityVector(units=1),
            required_labels=features,
            earliest_start=utc.start,
            latest_end=utc.end,
            request_id=request_id,
            eligible=eligible,
        )
        return MeetingBooking(placement, attendees, utc)

    def cancel(self, meeting_id: str, reservation_id: str) -> Placement:
        return self._engine.cancel(meeting_id, reservation_id)

    def snapshot(self) -> ReservationSnapshot:
        return self._engine.snapshot()


def create_meeting_room_scheduler(
    *, rooms: tuple[MeetingRoom, ...] | None = None
) -> MeetingRoomScheduler:
    """Construct a meeting-room scheduler."""
    return MeetingRoomScheduler(
        rooms
        or (
            MeetingRoom("room-a", 8, frozenset({"video"})),
        )
    )
