"""Airline gate assignment with compatibility and turnaround buffers."""

from __future__ import annotations

from dataclasses import dataclass

from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications._shared.reservations import ReservationSnapshot
from treemendous.applications.scheduling._common import Placement, names, text
from treemendous.applications.scheduling._placement import (
    BoundedPlacementEngine,
    LabeledResource,
)
from treemendous.domain import Span


@dataclass(frozen=True)
class Gate:
    """A gate and the aircraft type labels it can accept."""

    name: str
    aircraft_types: frozenset[str]

    def __post_init__(self) -> None:
        text(self.name, "gate name")
        names(self.aircraft_types, "aircraft_types")
        if not self.aircraft_types:
            raise ValueError("a gate must support at least one aircraft type")


class AirlineGateScheduler:
    """Assigns earliest gate/name deterministically, not network-wide optimally."""

    def __init__(self, gates: tuple[Gate, ...]) -> None:
        self._engine = BoundedPlacementEngine(
            tuple(
                LabeledResource(
                    gate.name,
                    CapacityVector(units=1),
                    gate.aircraft_types,
                )
                for gate in gates
            )
        )

    def assign(
        self,
        flight_id: str,
        arrival: int,
        departure: int,
        *,
        aircraft_type: str,
        turnaround_before: int = 0,
        turnaround_after: int = 0,
        request_id: str | None = None,
    ) -> Placement:
        service = Span(arrival, departure)
        text(aircraft_type, "aircraft_type")
        return self._engine.place(
            flight_id,
            service.length,
            CapacityVector(units=1),
            required_labels=frozenset({aircraft_type}),
            earliest_start=service.start,
            latest_end=service.end,
            request_id=request_id,
            buffer_before=turnaround_before,
            buffer_after=turnaround_after,
        )

    def cancel(self, flight_id: str, reservation_id: str) -> Placement:
        return self._engine.cancel(flight_id, reservation_id)

    def snapshot(self) -> ReservationSnapshot:
        return self._engine.snapshot()


def create_airline_gate_scheduler(
    *, gates: tuple[Gate, ...] | None = None
) -> AirlineGateScheduler:
    """Construct a gate scheduler."""
    return AirlineGateScheduler(gates or (Gate("gate-a", frozenset({"A320"})),))
