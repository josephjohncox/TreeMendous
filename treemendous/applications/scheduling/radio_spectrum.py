"""Exact channel-by-time rectangle reservations with guard bands."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from threading import RLock

from treemendous.applications.scheduling._common import integer, text
from treemendous.domain import Span
from treemendous.multidimensional import Box, BoxHandle, BoxIndex, BoxIndexSnapshot


class SpectrumStatus(str, Enum):
    ACTIVE = "active"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class SpectrumReservation:
    """One requested channel/time rectangle and its guard width."""

    reservation_id: str
    owner: str
    channel_start: int
    channel_end: int
    start: int
    end: int
    guard_channels: int
    request_id: str | None = None
    status: SpectrumStatus = SpectrumStatus.ACTIVE

    @property
    def id(self) -> str:
        return self.reservation_id

    @property
    def active(self) -> bool:
        return self.status is SpectrumStatus.ACTIVE


@dataclass(frozen=True)
class SpectrumConflict:
    requested: Box
    conflicting_ids: tuple[str, ...]


class SpectrumConflictError(ValueError):
    """Raised with identities of rectangles intersecting guarded geometry."""

    def __init__(self, conflict: SpectrumConflict) -> None:
        self.conflict = conflict
        super().__init__(
            "guarded channel/time rectangle overlaps "
            + ", ".join(conflict.conflicting_ids)
        )


@dataclass(frozen=True)
class SpectrumSnapshot:
    reservations: tuple[SpectrumReservation, ...]
    geometry: BoxIndexSnapshot


@dataclass(frozen=True)
class _SpectrumRecord:
    reservation: SpectrumReservation
    handle: BoxHandle | None
    box: Box


class RadioSpectrumScheduler:
    """Indexes exact integer rectangles; it is not an RF/interference optimizer."""

    def __init__(self, channel_count: int) -> None:
        integer(channel_count, "channel_count", minimum=1)
        self._channel_count = channel_count
        self._index = BoxIndex(2)
        self._records: dict[str, _SpectrumRecord] = {}
        self._next_by_owner: dict[str, int] = {}
        self._requests: dict[tuple[str, str], tuple[object, str]] = {}
        self._lock = RLock()

    def reserve(
        self,
        owner: str,
        channel_start: int,
        channel_width: int,
        start: int,
        end: int,
        *,
        guard_channels: int = 0,
        request_id: str | None = None,
    ) -> SpectrumReservation:
        text(owner, "owner")
        integer(channel_start, "channel_start")
        integer(channel_width, "channel_width", minimum=1)
        integer(guard_channels, "guard_channels")
        time = Span(start, end)
        channel_end = channel_start + channel_width
        if channel_end > self._channel_count:
            raise ValueError("requested channels exceed the managed channel domain")
        if request_id is not None:
            text(request_id, "request_id")
        fingerprint: object = (
            channel_start,
            channel_width,
            start,
            end,
            guard_channels,
        )
        guarded = Box(
            (max(0, channel_start - guard_channels), time.start),
            (min(self._channel_count, channel_end + guard_channels), time.end),
        )
        with self._lock:
            if request_id is not None:
                prior = self._requests.get((owner, request_id))
                if prior is not None:
                    if prior[0] != fingerprint:
                        raise ValueError(
                            "idempotency key was already used for a different request"
                        )
                    return self._records[prior[1]].reservation
            overlaps = self._index.overlaps(guarded)
            if overlaps:
                identifiers = tuple(
                    sorted(str(entry.data) for entry in overlaps)
                )
                raise SpectrumConflictError(SpectrumConflict(guarded, identifiers))
            sequence = self._next_by_owner.get(owner, 1)
            reservation_id = f"{owner}:{sequence}"
            reservation = SpectrumReservation(
                reservation_id,
                owner,
                channel_start,
                channel_end,
                time.start,
                time.end,
                guard_channels,
                request_id,
            )
            handle = self._index.insert(guarded, reservation_id)
            self._records[reservation_id] = _SpectrumRecord(
                reservation, handle, guarded
            )
            self._next_by_owner[owner] = sequence + 1
            if request_id is not None:
                self._requests[(owner, request_id)] = fingerprint, reservation_id
            return reservation

    def conflicts_for(
        self,
        channel_start: int,
        channel_width: int,
        start: int,
        end: int,
        *,
        guard_channels: int = 0,
    ) -> SpectrumConflict | None:
        integer(channel_start, "channel_start")
        integer(channel_width, "channel_width", minimum=1)
        integer(guard_channels, "guard_channels")
        time = Span(start, end)
        channel_end = channel_start + channel_width
        if channel_end > self._channel_count:
            raise ValueError("requested channels exceed the managed channel domain")
        box = Box(
            (max(0, channel_start - guard_channels), time.start),
            (min(self._channel_count, channel_end + guard_channels), time.end),
        )
        with self._lock:
            overlaps = self._index.overlaps(box)
            if not overlaps:
                return None
            return SpectrumConflict(
                box, tuple(sorted(str(entry.data) for entry in overlaps))
            )

    def cancel(self, owner: str, reservation_id: str) -> SpectrumReservation:
        text(owner, "owner")
        text(reservation_id, "reservation_id")
        with self._lock:
            record = self._records.get(reservation_id)
            if record is None:
                raise KeyError(reservation_id)
            if record.reservation.owner != owner:
                raise PermissionError("reservation belongs to a different owner")
            if not record.reservation.active:
                return record.reservation
            if record.handle is None:
                raise RuntimeError("active spectrum record has no geometry handle")
            self._index.remove(record.handle)
            cancelled = replace(
                record.reservation, status=SpectrumStatus.CANCELLED
            )
            self._records[reservation_id] = _SpectrumRecord(
                cancelled, None, record.box
            )
            return cancelled

    def snapshot(self) -> SpectrumSnapshot:
        with self._lock:
            reservations = tuple(
                sorted(
                    (record.reservation for record in self._records.values()),
                    key=lambda item: (item.start, item.channel_start, item.id),
                )
            )
            return SpectrumSnapshot(reservations, self._index.snapshot())


def create_radio_spectrum_scheduler(
    *, channel_count: int = 64
) -> RadioSpectrumScheduler:
    """Construct an exact rectangle spectrum scheduler."""
    return RadioSpectrumScheduler(channel_count)
