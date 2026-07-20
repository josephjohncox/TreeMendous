"""Priority-aware time-series alert and suppression windows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from treemendous.applications._shared.interval_records import (
    IntervalRecord,
    IntervalRecordIndex,
    IntervalRecordSnapshot,
    RecordHandle,
)


class WindowKind(str, Enum):
    """A firing alert or a suppression policy."""

    ALERT = "alert"
    SUPPRESSION = "suppression"


@dataclass(frozen=True)
class AlertWindow:
    """One series-local alert or suppression window."""

    window_id: str
    series: str
    kind: WindowKind
    priority: int
    label: str


AlertRecord = IntervalRecord[str, AlertWindow]
AlertHandle = RecordHandle[str]
AlertSnapshot = IntervalRecordSnapshot[str, AlertWindow]


@dataclass(frozen=True)
class AlertEvaluation:
    """Active policies and alerts after priority suppression."""

    alerts: tuple[AlertRecord, ...]
    suppressions: tuple[AlertRecord, ...]
    suppressed: tuple[AlertRecord, ...]


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _priority(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("priority must be an integer")
    return value


class AlertCatalog:
    """Evaluate active windows without merging coincident alert identities."""

    def __init__(self) -> None:
        self._index = IntervalRecordIndex[str, AlertWindow](lambda value: value)

    def add(
        self,
        window_id: str,
        start: int,
        end: int,
        *,
        series: str,
        kind: WindowKind | str,
        priority: int,
        label: str,
    ) -> AlertHandle:
        """Add one half-open active window."""
        try:
            parsed_kind = WindowKind(kind)
        except (TypeError, ValueError) as exc:
            raise ValueError("kind must be 'alert' or 'suppression'") from exc
        window = AlertWindow(
            _text(window_id, "window_id"),
            _text(series, "series"),
            parsed_kind,
            _priority(priority),
            _text(label, "label"),
        )
        return self._index.insert(window_id, start, end, window)

    def update(
        self,
        handle: AlertHandle,
        *,
        start: int | None = None,
        end: int | None = None,
        kind: WindowKind | str | None = None,
        priority: int | None = None,
        label: str | None = None,
    ) -> AlertRecord:
        """Update a policy window while retaining its handle."""
        current = self._index.get(handle)
        window = replace(
            current.payload,
            kind=current.payload.kind if kind is None else WindowKind(kind),
            priority=(
                current.payload.priority if priority is None else _priority(priority)
            ),
            label=current.payload.label if label is None else _text(label, "label"),
        )
        return self._index.update(
            handle, owner=handle.owner, start=start, end=end, payload=window
        )

    def remove(self, handle: AlertHandle) -> AlertRecord:
        """Remove one alert or suppression identity."""
        return self._index.remove(handle, owner=handle.owner)

    @staticmethod
    def _ordered(records: list[AlertRecord]) -> tuple[AlertRecord, ...]:
        return tuple(
            sorted(
                records,
                key=lambda record: (-record.payload.priority, record.insertion_order),
            )
        )

    def active_at(self, series: str, timestamp: int) -> AlertEvaluation:
        """Evaluate active alerts; equal/higher-priority suppressions win."""
        records = [
            record
            for record in self._index.at(timestamp)
            if record.payload.series == series
        ]
        suppressions = [
            record
            for record in records
            if record.payload.kind is WindowKind.SUPPRESSION
        ]
        threshold = max(
            (record.payload.priority for record in suppressions), default=None
        )
        candidates = [
            record for record in records if record.payload.kind is WindowKind.ALERT
        ]
        alerts = [
            record
            for record in candidates
            if threshold is None or record.payload.priority > threshold
        ]
        suppressed = [record for record in candidates if record not in alerts]
        return AlertEvaluation(
            self._ordered(alerts),
            self._ordered(suppressions),
            self._ordered(suppressed),
        )

    def active_windows(
        self, series: str, start: int, end: int
    ) -> tuple[AlertRecord, ...]:
        """Return all intersecting policy windows by priority and insertion."""
        records = [
            record
            for record in self._index.overlaps(start, end)
            if record.payload.series == series
        ]
        return self._ordered(records)

    def snapshot(self) -> AlertSnapshot:
        """Return an immutable insertion-ordered snapshot."""
        return self._index.snapshot()


def create_catalog() -> AlertCatalog:
    """Create an empty alert catalog."""
    return AlertCatalog()
