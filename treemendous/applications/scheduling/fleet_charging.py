"""Bounded deterministic fleet charging-window scheduler."""

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
    names,
    text,
)
from treemendous.domain import Span


@dataclass(frozen=True)
class Charger:
    """An exclusive charger delivering integer energy units per slot."""

    name: str
    power_per_slot: int
    connectors: frozenset[str]

    def __post_init__(self) -> None:
        text(self.name, "charger name")
        integer(self.power_per_slot, "power_per_slot", minimum=1)
        names(self.connectors, "connectors")
        if not self.connectors:
            raise ValueError("a charger must support at least one connector")


@dataclass(frozen=True)
class ChargingSession:
    charger: str
    energy: int
    power_per_slot: int
    reservation: Reservation

    @property
    def duration(self) -> int:
        return self.reservation.end - self.reservation.start

    @property
    def id(self) -> str:
        return self.reservation.id


class FleetChargingScheduler:
    """Selects earliest completion within dwell and a configured session bound.

    Energy and power use caller-defined integer units.  This policy ignores
    tariffs, taper curves, battery degradation, and grid-wide optimization.
    """

    def __init__(
        self, chargers: tuple[Charger, ...], *, max_session_slots: int
    ) -> None:
        if not chargers:
            raise ValueError("at least one charger is required")
        integer(max_session_slots, "max_session_slots", minimum=1)
        self._chargers = {charger.name: charger for charger in chargers}
        if len(self._chargers) != len(chargers):
            raise ValueError("charger names must be unique")
        self._max_session_slots = max_session_slots
        self._ledger = ReservationLedger(
            {name: CapacityVector(units=1) for name in self._chargers}
        )
        self._requests: dict[tuple[str, str], tuple[object, ChargingSession]] = {}
        self._sessions: dict[str, ChargingSession] = {}
        self._lock = RLock()

    def schedule(
        self,
        vehicle_id: str,
        energy: int,
        *,
        connector: str,
        arrival: int,
        departure: int,
        request_id: str | None = None,
    ) -> ChargingSession:
        text(vehicle_id, "vehicle_id")
        integer(energy, "energy", minimum=1)
        text(connector, "connector")
        dwell = Span(arrival, departure)
        if request_id is not None:
            text(request_id, "request_id")
        fingerprint: object = (energy, connector, arrival, departure)
        with self._lock:
            if request_id is not None:
                prior = self._requests.get((vehicle_id, request_id))
                if prior is not None:
                    if prior[0] != fingerprint:
                        raise ValueError(
                            "idempotency key was already used for a different request"
                        )
                    return prior[1]
            candidates: list[tuple[int, int, str, Charger]] = []
            last_conflicts: tuple[ReservationConflict, ...] = ()
            compatible = tuple(
                charger
                for charger in sorted(
                    self._chargers.values(), key=lambda item: item.name
                )
                if connector in charger.connectors
            )
            for charger in compatible:
                duration = (
                    energy + charger.power_per_slot - 1
                ) // charger.power_per_slot
                if duration > self._max_session_slots or duration > dwell.length:
                    continue
                for start in range(dwell.start, dwell.end - duration + 1):
                    conflicts = self._ledger.conflicts_for(
                        start,
                        start + duration,
                        {charger.name: CapacityVector(units=1)},
                    )
                    if not conflicts:
                        candidates.append(
                            (start + duration, start, charger.name, charger)
                        )
                        break
                    last_conflicts = conflicts
            if not candidates:
                raise SchedulingUnavailableError(
                    "energy cannot be delivered by a compatible charger within dwell",
                    conflicts=last_conflicts,
                    considered=tuple(item.name for item in compatible),
                )
            end, start, _, charger = min(candidates)
            reservation = self._ledger.reserve_exact(
                vehicle_id,
                start,
                end,
                {charger.name: CapacityVector(units=1)},
                request_id=request_id,
            )
            session = ChargingSession(
                charger.name, energy, charger.power_per_slot, reservation
            )
            self._sessions[session.id] = session
            if request_id is not None:
                self._requests[(vehicle_id, request_id)] = fingerprint, session
            return session

    def cancel(self, vehicle_id: str, reservation_id: str) -> ChargingSession:
        with self._lock:
            reservation = self._ledger.cancel(vehicle_id, reservation_id)
            prior = self._sessions[reservation_id]
            cancelled = ChargingSession(
                prior.charger,
                prior.energy,
                prior.power_per_slot,
                reservation,
            )
            self._sessions[reservation_id] = cancelled
            for key, (fingerprint, session) in tuple(self._requests.items()):
                if session.id == reservation_id:
                    self._requests[key] = fingerprint, cancelled
            return cancelled

    def snapshot(self) -> ReservationSnapshot:
        return self._ledger.snapshot()


def create_fleet_charging_scheduler(
    *,
    chargers: tuple[Charger, ...] | None = None,
    max_session_slots: int = 24,
) -> FleetChargingScheduler:
    """Construct a bounded fleet charging scheduler."""
    return FleetChargingScheduler(
        chargers or (Charger("charger-a", 10, frozenset({"ccs"})),),
        max_session_slots=max_session_slots,
    )
