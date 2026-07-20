"""Maintenance planning with dependency, blackout, and service-window validation."""

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
    integer,
    positive,
    spans,
    text,
)
from treemendous.domain import Span, validate_coordinate


@dataclass(frozen=True)
class MaintenanceService:
    """A service's allowed windows, prohibited blackouts, and concurrency."""

    name: str
    service_windows: tuple[Span, ...]
    blackouts: tuple[Span, ...] = ()
    concurrency: int = 1

    def __post_init__(self) -> None:
        text(self.name, "service name")
        spans(self.service_windows, "service_windows")
        if not isinstance(self.blackouts, tuple) or not all(
            isinstance(item, Span) for item in self.blackouts
        ):
            raise TypeError("blackouts must be a tuple of Span values")
        integer(self.concurrency, "concurrency", minimum=1)


@dataclass(frozen=True)
class MaintenanceBooking:
    task_id: str
    service: str
    dependencies: tuple[str, ...]
    reservation: Reservation

    @property
    def id(self) -> str:
        return self.reservation.id


class MaintenanceScheduler:
    """Schedules dependency-ready tasks; it is not a project optimizer."""

    def __init__(self, services: tuple[MaintenanceService, ...]) -> None:
        if not services:
            raise ValueError("at least one maintenance service is required")
        self._services = {service.name: service for service in services}
        if len(self._services) != len(services):
            raise ValueError("service names must be unique")
        self._ledger = ReservationLedger(
            {
                name: CapacityVector(slots=service.concurrency)
                for name, service in self._services.items()
            }
        )
        self._tasks: dict[str, MaintenanceBooking] = {}
        self._requests: dict[tuple[str, str], tuple[object, MaintenanceBooking]] = {}
        self._lock = RLock()

    def schedule(
        self,
        task_id: str,
        service: str,
        duration: int,
        *,
        dependencies: tuple[str, ...] = (),
        earliest_start: int = 0,
        latest_end: int,
        request_id: str | None = None,
    ) -> MaintenanceBooking:
        text(task_id, "task_id")
        text(service, "service")
        positive(duration, "duration")
        validate_coordinate(earliest_start, "earliest_start")
        validate_coordinate(latest_end, "latest_end")
        if service not in self._services:
            raise KeyError(f"unknown service: {service!r}")
        if task_id in dependencies or len(set(dependencies)) != len(dependencies):
            raise ValueError("dependencies must be unique and cannot include the task")
        for dependency in dependencies:
            text(dependency, "dependency")
        if request_id is not None:
            text(request_id, "request_id")
        fingerprint: object = (
            service,
            duration,
            tuple(sorted(dependencies)),
            earliest_start,
            latest_end,
        )
        with self._lock:
            if request_id is not None:
                prior = self._requests.get((task_id, request_id))
                if prior is not None:
                    if prior[0] != fingerprint:
                        raise ValueError(
                            "idempotency key was already used for a different request"
                        )
                    return prior[1]
            if task_id in self._tasks:
                raise ValueError("task_id is already scheduled")
            dependency_bookings: list[MaintenanceBooking] = []
            for dependency in dependencies:
                try:
                    booking = self._tasks[dependency]
                except KeyError:
                    raise KeyError(f"unknown dependency: {dependency!r}") from None
                if not booking.reservation.active:
                    raise ValueError(f"dependency is cancelled: {dependency!r}")
                dependency_bookings.append(booking)
            ready = max(
                (earliest_start, *(item.reservation.end for item in dependency_bookings))
            )
            policy = self._services[service]
            selected: int | None = None
            last_conflicts: tuple[ReservationConflict, ...] = ()
            for start in range(ready, latest_end - duration + 1):
                candidate = Span(start, start + duration)
                if not any(
                    window.contains(candidate) for window in policy.service_windows
                ):
                    continue
                if any(blackout.overlaps(candidate) for blackout in policy.blackouts):
                    continue
                conflicts = self._ledger.conflicts_for(
                    candidate.start,
                    candidate.end,
                    {service: CapacityVector(slots=1)},
                )
                if not conflicts:
                    selected = start
                    break
                last_conflicts = conflicts
            if selected is None:
                raise SchedulingUnavailableError(
                    "no dependency-ready service window avoids blackouts",
                    conflicts=last_conflicts,
                    considered=(service,),
                )
            reservation = self._ledger.reserve_exact(
                task_id,
                selected,
                selected + duration,
                {service: CapacityVector(slots=1)},
                request_id=request_id,
            )
            booking = MaintenanceBooking(
                task_id, service, tuple(sorted(dependencies)), reservation
            )
            self._tasks[task_id] = booking
            if request_id is not None:
                self._requests[(task_id, request_id)] = fingerprint, booking
            return booking

    def cancel(self, task_id: str) -> MaintenanceBooking:
        with self._lock:
            try:
                booking = self._tasks[task_id]
            except KeyError:
                raise KeyError(task_id) from None
            reservation = self._ledger.cancel(task_id, booking.id)
            cancelled = MaintenanceBooking(
                booking.task_id,
                booking.service,
                booking.dependencies,
                reservation,
            )
            self._tasks[task_id] = cancelled
            for key, (fingerprint, prior) in tuple(self._requests.items()):
                if prior.task_id == task_id:
                    self._requests[key] = fingerprint, cancelled
            return cancelled

    def snapshot(self) -> ReservationSnapshot:
        return self._ledger.snapshot()


def create_maintenance_scheduler(
    *, services: tuple[MaintenanceService, ...] | None = None
) -> MaintenanceScheduler:
    """Construct a maintenance scheduler."""
    return MaintenanceScheduler(
        services
        or (
            MaintenanceService("api", (Span(0, 100),)),
        )
    )
