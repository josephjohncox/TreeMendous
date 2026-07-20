"""Warehouse dock appointments with cargo compatibility and handling buffers."""

from __future__ import annotations

from dataclasses import dataclass

from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications._shared.reservations import ReservationSnapshot
from treemendous.applications.scheduling._common import Placement, names, text
from treemendous.applications.scheduling._placement import (
    BoundedPlacementEngine,
    LabeledResource,
)


@dataclass(frozen=True)
class Dock:
    """A dock supporting a nonempty set of cargo labels."""

    name: str
    cargo_types: frozenset[str]

    def __post_init__(self) -> None:
        text(self.name, "dock name")
        names(self.cargo_types, "cargo_types")
        if not self.cargo_types:
            raise ValueError("a dock must support at least one cargo type")


class WarehouseDockScheduler:
    """Places bounded appointments with pre/post handling occupancy."""

    def __init__(self, docks: tuple[Dock, ...]) -> None:
        self._engine = BoundedPlacementEngine(
            tuple(
                LabeledResource(
                    dock.name,
                    CapacityVector(units=1),
                    dock.cargo_types,
                )
                for dock in docks
            )
        )

    def book(
        self,
        carrier_id: str,
        duration: int,
        *,
        cargo_type: str,
        earliest_start: int,
        latest_end: int,
        handling_before: int = 0,
        handling_after: int = 0,
        request_id: str | None = None,
    ) -> Placement:
        text(cargo_type, "cargo_type")
        return self._engine.place(
            carrier_id,
            duration,
            CapacityVector(units=1),
            required_labels=frozenset({cargo_type}),
            earliest_start=earliest_start,
            latest_end=latest_end,
            request_id=request_id,
            buffer_before=handling_before,
            buffer_after=handling_after,
        )

    def cancel(self, carrier_id: str, reservation_id: str) -> Placement:
        return self._engine.cancel(carrier_id, reservation_id)

    def snapshot(self) -> ReservationSnapshot:
        return self._engine.snapshot()


def create_warehouse_dock_scheduler(
    *, docks: tuple[Dock, ...] | None = None
) -> WarehouseDockScheduler:
    """Construct a dock scheduler."""
    return WarehouseDockScheduler(docks or (Dock("dock-a", frozenset({"dry"})),))
