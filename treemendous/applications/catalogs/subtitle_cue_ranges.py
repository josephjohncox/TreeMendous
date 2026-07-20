"""Language- and layer-aware subtitle cue interval catalog."""

from __future__ import annotations

from dataclasses import dataclass, replace

from treemendous.applications._shared.interval_records import (
    IntervalRecord,
    IntervalRecordIndex,
    IntervalRecordSnapshot,
    RecordHandle,
)


@dataclass(frozen=True)
class SubtitleCue:
    """One retained subtitle cue identity and rendering metadata."""

    cue_id: str
    language: str
    layer: int
    text: str


CueRecord = IntervalRecord[str, SubtitleCue]
CueHandle = RecordHandle[str]
SubtitleSnapshot = IntervalRecordSnapshot[str, SubtitleCue]


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _layer(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("layer must be a nonnegative integer")
    return value


class SubtitleCatalog:
    """Keep coincident cues separate and return explicit rendering order."""

    def __init__(self) -> None:
        self._index = IntervalRecordIndex[str, SubtitleCue](lambda value: value)

    def add(
        self,
        cue_id: str,
        start: int,
        end: int,
        *,
        language: str,
        layer: int,
        text: str,
    ) -> CueHandle:
        """Add a half-open cue interval."""
        cue = SubtitleCue(
            _text(cue_id, "cue_id"),
            _text(language, "language"),
            _layer(layer),
            _text(text, "text"),
        )
        return self._index.insert(cue_id, start, end, cue)

    def update(
        self,
        handle: CueHandle,
        *,
        start: int | None = None,
        end: int | None = None,
        language: str | None = None,
        layer: int | None = None,
        text: str | None = None,
    ) -> CueRecord:
        """Update a cue while retaining its stable handle."""
        current = self._index.get(handle)
        cue = replace(
            current.payload,
            language=(
                current.payload.language
                if language is None
                else _text(language, "language")
            ),
            layer=current.payload.layer if layer is None else _layer(layer),
            text=current.payload.text if text is None else _text(text, "text"),
        )
        return self._index.update(
            handle, owner=handle.owner, start=start, end=end, payload=cue
        )

    def remove(self, handle: CueHandle) -> CueRecord:
        """Remove one cue identity."""
        return self._index.remove(handle, owner=handle.owner)

    @staticmethod
    def _ordered(records: tuple[CueRecord, ...]) -> tuple[CueRecord, ...]:
        return tuple(
            sorted(
                records,
                key=lambda record: (
                    record.payload.layer,
                    record.start,
                    record.insertion_order,
                ),
            )
        )

    def active_at(
        self, time: int, *, language: str | None = None
    ) -> tuple[CueRecord, ...]:
        """Return active cues bottom-layer first, then start/insertion order."""
        records = tuple(
            record
            for record in self._index.at(time)
            if language is None or record.payload.language == language
        )
        return self._ordered(records)

    def in_window(
        self, start: int, end: int, *, language: str | None = None
    ) -> tuple[CueRecord, ...]:
        """Return cues intersecting an edit/playback window in render order."""
        records = tuple(
            record
            for record in self._index.overlaps(start, end)
            if language is None or record.payload.language == language
        )
        return self._ordered(records)

    def snapshot(self) -> SubtitleSnapshot:
        """Return an immutable insertion-ordered snapshot."""
        return self._index.snapshot()


def create_catalog() -> SubtitleCatalog:
    """Create an empty subtitle cue catalog."""
    return SubtitleCatalog()
