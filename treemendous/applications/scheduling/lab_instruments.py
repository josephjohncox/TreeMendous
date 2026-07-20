"""Laboratory instrument booking with capability, calibration, and cleanup rules."""

from __future__ import annotations

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
    integer,
    names,
    positive,
    spans,
    text,
)
from treemendous.domain import Span, validate_coordinate


@dataclass(frozen=True)
class LabInstrument:
    """An exclusive instrument and its calibrated service windows."""

    name: str
    capabilities: frozenset[str]
    calibration_windows: tuple[Span, ...]
    cleanup_slots: int = 0

    def __post_init__(self) -> None:
        text(self.name, "instrument name")
        names(self.capabilities, "instrument capabilities")
        spans(self.calibration_windows, "calibration_windows")
        integer(self.cleanup_slots, "cleanup_slots")


class LabInstrumentScheduler:
    """Chooses earliest calibrated integer slot then instrument name."""

    def __init__(self, instruments: tuple[LabInstrument, ...]) -> None:
        if not instruments:
            raise ValueError("at least one instrument is required")
        self._instruments = {item.name: item for item in instruments}
        if len(self._instruments) != len(instruments):
            raise ValueError("instrument names must be unique")
        self._ledger = ReservationLedger(
            {name: CapacityVector(units=1) for name in self._instruments}
        )
        self._requests: dict[tuple[str, str], tuple[object, Placement]] = {}
        self._lock = RLock()

    def book(
        self,
        experiment_id: str,
        duration: int,
        *,
        capabilities: frozenset[str],
        earliest_start: int,
        latest_end: int,
        request_id: str | None = None,
    ) -> Placement:
        text(experiment_id, "experiment_id")
        positive(duration, "duration")
        names(capabilities, "capabilities")
        validate_coordinate(earliest_start, "earliest_start")
        validate_coordinate(latest_end, "latest_end")
        if request_id is not None:
            text(request_id, "request_id")
        fingerprint: object = (
            duration,
            capabilities,
            earliest_start,
            latest_end,
        )
        with self._lock:
            if request_id is not None:
                prior = self._requests.get((experiment_id, request_id))
                if prior is not None:
                    if prior[0] != fingerprint:
                        raise ValueError(
                            "idempotency key was already used for a different request"
                        )
                    return prior[1]
            compatible = tuple(
                item
                for item in sorted(
                    self._instruments.values(), key=lambda candidate: candidate.name
                )
                if capabilities.issubset(item.capabilities)
            )
            if not compatible:
                raise SchedulingUnavailableError("no instrument has the capabilities")
            selected: tuple[int, LabInstrument] | None = None
            last_conflicts: tuple[ReservationConflict, ...] = ()
            for start in range(earliest_start, latest_end - duration + 1):
                for instrument in compatible:
                    occupied_end = start + duration + instrument.cleanup_slots
                    if not any(
                        window.start <= start and occupied_end <= window.end
                        for window in instrument.calibration_windows
                    ):
                        continue
                    conflicts = self._ledger.conflicts_for(
                        start,
                        start + duration,
                        {instrument.name: CapacityVector(units=1)},
                        buffer_after=instrument.cleanup_slots,
                    )
                    if not conflicts:
                        selected = start, instrument
                        break
                    last_conflicts = conflicts
                if selected is not None:
                    break
            if selected is None:
                raise SchedulingUnavailableError(
                    "no calibrated instrument slot is available",
                    conflicts=last_conflicts,
                    considered=tuple(item.name for item in compatible),
                )
            start, instrument = selected
            reservation = self._ledger.reserve_exact(
                experiment_id,
                start,
                start + duration,
                {instrument.name: CapacityVector(units=1)},
                request_id=request_id,
                buffer_after=instrument.cleanup_slots,
            )
            placement = Placement(instrument.name, reservation)
            if request_id is not None:
                self._requests[(experiment_id, request_id)] = fingerprint, placement
            return placement

    def cancel(self, experiment_id: str, reservation_id: str) -> Placement:
        with self._lock:
            reservation = self._ledger.cancel(experiment_id, reservation_id)
            placement = Placement(reservation.requirements[0].resource, reservation)
            for key, (fingerprint, prior) in tuple(self._requests.items()):
                if prior.id == reservation_id:
                    self._requests[key] = fingerprint, placement
            return placement

    def snapshot(self) -> ReservationSnapshot:
        return self._ledger.snapshot()


def create_lab_instrument_scheduler(
    *, instruments: tuple[LabInstrument, ...] | None = None
) -> LabInstrumentScheduler:
    """Construct a laboratory instrument scheduler."""
    return LabInstrumentScheduler(
        instruments
        or (
            LabInstrument(
                "scope-a",
                frozenset({"imaging"}),
                (Span(0, 100),),
                cleanup_slots=2,
            ),
        )
    )
