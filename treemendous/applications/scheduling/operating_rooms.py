"""Operating-room scheduling with atomic room/staff/equipment reservations."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications._shared.reservations import (
    Reservation,
    ReservationConflict,
    ReservationLedger,
    ReservationSnapshot,
)
from treemendous.applications.scheduling._common import (
    SchedulingUnavailableError,
    names,
    positive,
    text,
)
from treemendous.domain import validate_coordinate


@dataclass(frozen=True)
class OperatingRoom:
    name: str
    capabilities: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        text(self.name, "room name")
        names(self.capabilities, "room capabilities")


@dataclass(frozen=True)
class ClinicalResource:
    """A named exclusive staff member or equipment item."""

    name: str

    def __post_init__(self) -> None:
        text(self.name, "clinical resource name")


@dataclass(frozen=True)
class ProcedureBooking:
    room: str
    staff: tuple[str, ...]
    equipment: tuple[str, ...]
    reservation: Reservation

    @property
    def id(self) -> str:
        return self.reservation.id


class OperatingRoomScheduler:
    """Commits room, staff, and equipment in one ledger transition."""

    def __init__(
        self,
        rooms: tuple[OperatingRoom, ...],
        staff: tuple[ClinicalResource, ...],
        equipment: tuple[ClinicalResource, ...],
    ) -> None:
        if not rooms:
            raise ValueError("at least one operating room is required")
        self._rooms = {room.name: room for room in rooms}
        self._staff = {item.name for item in staff}
        self._equipment = {item.name for item in equipment}
        if len(self._rooms) != len(rooms):
            raise ValueError("operating room names must be unique")
        if len(self._staff) != len(staff) or len(self._equipment) != len(equipment):
            raise ValueError("clinical resource names must be unique by kind")
        resources = {f"room:{name}": CapacityVector(units=1) for name in self._rooms}
        resources.update(
            {f"staff:{name}": CapacityVector(units=1) for name in self._staff}
        )
        resources.update(
            {f"equipment:{name}": CapacityVector(units=1) for name in self._equipment}
        )
        self._ledger = ReservationLedger(resources)
        self._requests: dict[tuple[str, str], tuple[object, ProcedureBooking]] = {}
        self._lock = RLock()

    def schedule(
        self,
        procedure_id: str,
        duration: int,
        *,
        room_capabilities: frozenset[str] = frozenset(),
        staff: tuple[str, ...],
        equipment: tuple[str, ...],
        earliest_start: int,
        latest_end: int,
        request_id: str | None = None,
    ) -> ProcedureBooking:
        text(procedure_id, "procedure_id")
        positive(duration, "duration")
        names(room_capabilities, "room_capabilities")
        validate_coordinate(earliest_start, "earliest_start")
        validate_coordinate(latest_end, "latest_end")
        if len(set(staff)) != len(staff) or len(set(equipment)) != len(equipment):
            raise ValueError("staff and equipment requests must be unique")
        for item in staff:
            text(item, "staff")
            if item not in self._staff:
                raise KeyError(f"unknown staff resource: {item!r}")
        for item in equipment:
            text(item, "equipment")
            if item not in self._equipment:
                raise KeyError(f"unknown equipment resource: {item!r}")
        if request_id is not None:
            text(request_id, "request_id")
        fingerprint: object = (
            duration,
            room_capabilities,
            tuple(sorted(staff)),
            tuple(sorted(equipment)),
            earliest_start,
            latest_end,
        )
        with self._lock:
            if request_id is not None:
                prior = self._requests.get((procedure_id, request_id))
                if prior is not None:
                    if prior[0] != fingerprint:
                        raise ValueError(
                            "idempotency key was already used for a different request"
                        )
                    return prior[1]
            compatible = tuple(
                room.name
                for room in sorted(self._rooms.values(), key=lambda item: item.name)
                if room_capabilities.issubset(room.capabilities)
            )
            if not compatible:
                raise SchedulingUnavailableError("no operating room is compatible")
            selected: tuple[int, str] | None = None
            last_conflicts: tuple[ReservationConflict, ...] = ()
            for start in range(earliest_start, latest_end - duration + 1):
                for room in compatible:
                    requirements = self._requirements(room, staff, equipment)
                    conflicts = self._ledger.conflicts_for(
                        start, start + duration, requirements
                    )
                    if not conflicts:
                        selected = start, room
                        break
                    last_conflicts = conflicts
                if selected is not None:
                    break
            if selected is None:
                raise SchedulingUnavailableError(
                    "room/staff/equipment are not jointly available",
                    conflicts=last_conflicts,
                    considered=compatible,
                )
            start, room = selected
            reservation = self._ledger.reserve_exact(
                procedure_id,
                start,
                start + duration,
                self._requirements(room, staff, equipment),
                request_id=request_id,
            )
            booking = ProcedureBooking(
                room,
                tuple(sorted(staff)),
                tuple(sorted(equipment)),
                reservation,
            )
            if request_id is not None:
                self._requests[(procedure_id, request_id)] = fingerprint, booking
            return booking

    @staticmethod
    def _requirements(
        room: str, staff: tuple[str, ...], equipment: tuple[str, ...]
    ) -> dict[str, CapacityVector]:
        requirements = {f"room:{room}": CapacityVector(units=1)}
        requirements.update(
            {f"staff:{name}": CapacityVector(units=1) for name in staff}
        )
        requirements.update(
            {f"equipment:{name}": CapacityVector(units=1) for name in equipment}
        )
        return requirements

    def cancel(self, procedure_id: str, reservation_id: str) -> ProcedureBooking:
        with self._lock:
            reservation = self._ledger.cancel(procedure_id, reservation_id)
            room = next(
                item.resource.removeprefix("room:")
                for item in reservation.requirements
                if item.resource.startswith("room:")
            )
            staff = tuple(
                item.resource.removeprefix("staff:")
                for item in reservation.requirements
                if item.resource.startswith("staff:")
            )
            equipment = tuple(
                item.resource.removeprefix("equipment:")
                for item in reservation.requirements
                if item.resource.startswith("equipment:")
            )
            booking = ProcedureBooking(room, staff, equipment, reservation)
            for key, (fingerprint, prior) in tuple(self._requests.items()):
                if prior.id == reservation_id:
                    self._requests[key] = fingerprint, booking
            return booking

    def snapshot(self) -> ReservationSnapshot:
        return self._ledger.snapshot()


def create_operating_room_scheduler(
    *,
    rooms: tuple[OperatingRoom, ...] | None = None,
    staff: tuple[ClinicalResource, ...] | None = None,
    equipment: tuple[ClinicalResource, ...] | None = None,
) -> OperatingRoomScheduler:
    """Construct an operating-room scheduler."""
    return OperatingRoomScheduler(
        rooms or (OperatingRoom("or-1", frozenset({"general"})),),
        staff or (ClinicalResource("surgeon-1"),),
        equipment or (ClinicalResource("monitor-1"),),
    )
