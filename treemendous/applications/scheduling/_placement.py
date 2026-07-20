"""Private bounded placement kernel used by labeled scheduling scenarios."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock

from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications._shared.reservations import (
    ReservationConflict,
    ReservationLedger,
    ReservationSnapshot,
)
from treemendous.applications.scheduling._common import (
    Placement,
    SchedulingUnavailableError,
    names,
    positive,
    text,
)
from treemendous.domain import validate_coordinate


@dataclass(frozen=True)
class LabeledResource:
    """Named capacity-bearing resource with exact-match capability labels."""

    name: str
    capacity: CapacityVector
    labels: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        text(self.name, "resource name")
        if not isinstance(self.capacity, CapacityVector):
            raise TypeError("capacity must be a CapacityVector")
        names(self.labels, "labels")


class BoundedPlacementEngine:
    """Deterministic earliest placement over compatible labeled resources.

    The scan is intentionally bounded and deterministic, not an optimal packing
    algorithm.  A lock covers policy selection, ledger commit, and idempotency.
    """

    def __init__(self, resources: tuple[LabeledResource, ...]) -> None:
        if not resources:
            raise ValueError("at least one resource is required")
        by_name = {resource.name: resource for resource in resources}
        if len(by_name) != len(resources):
            raise ValueError("resource names must be unique")
        self._resources = dict(sorted(by_name.items()))
        self._ledger = ReservationLedger(
            {name: resource.capacity for name, resource in self._resources.items()}
        )
        self._requests: dict[tuple[str, str], tuple[object, Placement]] = {}
        self._lock = RLock()

    def place(
        self,
        owner: str,
        duration: int,
        demand: CapacityVector,
        *,
        required_labels: frozenset[str] = frozenset(),
        earliest_start: int = 0,
        latest_end: int,
        request_id: str | None = None,
        eligible: frozenset[str] | None = None,
        buffer_before: int = 0,
        buffer_after: int = 0,
    ) -> Placement:
        text(owner, "owner")
        positive(duration, "duration")
        if not isinstance(demand, CapacityVector):
            raise TypeError("demand must be a CapacityVector")
        names(required_labels, "required_labels")
        validate_coordinate(earliest_start, "earliest_start")
        validate_coordinate(latest_end, "latest_end")
        if earliest_start + duration > latest_end:
            raise ValueError("duration does not fit within the bounded window")
        if request_id is not None:
            text(request_id, "request_id")
        if eligible is not None:
            names(eligible, "eligible")
        fingerprint: object = (
            duration,
            demand,
            required_labels,
            earliest_start,
            latest_end,
            eligible,
            buffer_before,
            buffer_after,
        )
        with self._lock:
            if request_id is not None:
                prior = self._requests.get((owner, request_id))
                if prior is not None:
                    if prior[0] != fingerprint:
                        raise ValueError(
                            "idempotency key was already used for a different request"
                        )
                    return prior[1]

            compatible: list[str] = []
            for name, resource in self._resources.items():
                if eligible is not None and name not in eligible:
                    continue
                if not required_labels.issubset(resource.labels):
                    continue
                try:
                    resource.capacity._require_same_dimensions(demand)
                except ValueError:
                    continue
                if resource.capacity.fits(demand):
                    compatible.append(name)
            if not compatible:
                raise SchedulingUnavailableError(
                    "no resource satisfies labels and total capacity",
                    considered=tuple(self._resources),
                )

            chosen: tuple[int, str] | None = None
            diagnostics: tuple[ReservationConflict, ...] = ()
            for start in range(earliest_start, latest_end - duration + 1):
                for name in compatible:
                    conflicts = self._ledger.conflicts_for(
                        start,
                        start + duration,
                        {name: demand},
                        buffer_before=buffer_before,
                        buffer_after=buffer_after,
                    )
                    if not conflicts:
                        chosen = start, name
                        break
                    diagnostics = conflicts
                if chosen is not None:
                    break
            if chosen is None:
                raise SchedulingUnavailableError(
                    "compatible resources have no capacity in the bounded window",
                    conflicts=diagnostics,
                    considered=tuple(compatible),
                )
            start, resource_name = chosen
            reservation = self._ledger.reserve_exact(
                owner,
                start,
                start + duration,
                {resource_name: demand},
                request_id=request_id,
                buffer_before=buffer_before,
                buffer_after=buffer_after,
            )
            placement = Placement(resource_name, reservation)
            if request_id is not None:
                self._requests[(owner, request_id)] = fingerprint, placement
            return placement

    def cancel(self, owner: str, reservation_id: str) -> Placement:
        with self._lock:
            cancelled = self._ledger.cancel(owner, reservation_id)
            resource = cancelled.requirements[0].resource
            placement = Placement(resource, cancelled)
            for key, (fingerprint, prior) in tuple(self._requests.items()):
                if prior.id == reservation_id:
                    self._requests[key] = fingerprint, placement
            return placement

    def snapshot(self) -> ReservationSnapshot:
        return self._ledger.snapshot()

    def conflicts_for(
        self,
        start: int,
        end: int,
        requirements: Mapping[str, CapacityVector],
    ) -> tuple[ReservationConflict, ...]:
        return self._ledger.conflicts_for(start, end, requirements)
