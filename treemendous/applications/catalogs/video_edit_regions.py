"""Track/effect-preserving video edit region and invalidation catalog."""

from __future__ import annotations

from dataclasses import dataclass, replace

from treemendous.applications._shared.coverage import (
    CoverageSnapshot,
    coverage_snapshot,
)
from treemendous.applications._shared.interval_records import (
    IntervalRecord,
    IntervalRecordIndex,
    IntervalRecordSnapshot,
    RecordHandle,
)


@dataclass(frozen=True)
class EditRegion:
    """One edit operation attached to a track and effect kind."""

    region_id: str
    track: str
    effect: str
    parameters: tuple[tuple[str, str], ...]


EditRecord = IntervalRecord[str, EditRegion]
EditHandle = RecordHandle[str]
VideoEditSnapshot = IntervalRecordSnapshot[str, EditRegion]


@dataclass(frozen=True)
class Invalidation:
    """Identity-preserving invalidation records and derived frame coverage."""

    records: tuple[EditRecord, ...]
    coverage: CoverageSnapshot[str]


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _parameters(values: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    if values is None:
        return ()
    if not isinstance(values, dict) or not all(
        isinstance(key, str) and key and isinstance(value, str)
        for key, value in values.items()
    ):
        raise ValueError("parameters must map nonempty strings to strings")
    return tuple(sorted(values.items()))


class VideoEditCatalog:
    """Retain overlapping edits while deriving exact invalidation coverage."""

    def __init__(self) -> None:
        self._index = IntervalRecordIndex[str, EditRegion](lambda value: value)

    def add(
        self,
        region_id: str,
        start_frame: int,
        end_frame: int,
        *,
        track: str,
        effect: str,
        parameters: dict[str, str] | None = None,
    ) -> EditHandle:
        """Add one half-open frame edit region."""
        region = EditRegion(
            _text(region_id, "region_id"),
            _text(track, "track"),
            _text(effect, "effect"),
            _parameters(parameters),
        )
        return self._index.insert(region_id, start_frame, end_frame, region)

    def update(
        self,
        handle: EditHandle,
        *,
        start_frame: int | None = None,
        end_frame: int | None = None,
        track: str | None = None,
        effect: str | None = None,
        parameters: dict[str, str] | None = None,
    ) -> EditRecord:
        """Update an edit while preserving identity and original ordering."""
        current = self._index.get(handle)
        region = replace(
            current.payload,
            track=current.payload.track if track is None else _text(track, "track"),
            effect=(
                current.payload.effect if effect is None else _text(effect, "effect")
            ),
            parameters=(
                current.payload.parameters
                if parameters is None
                else _parameters(parameters)
            ),
        )
        return self._index.update(
            handle,
            owner=handle.owner,
            start=start_frame,
            end=end_frame,
            payload=region,
        )

    def remove(self, handle: EditHandle) -> EditRecord:
        """Remove one edit identity."""
        return self._index.remove(handle, owner=handle.owner)

    def regions(
        self,
        start_frame: int,
        end_frame: int,
        *,
        tracks: frozenset[str] | None = None,
        effects: frozenset[str] | None = None,
    ) -> tuple[EditRecord, ...]:
        """Return intersecting track/effect records in insertion order."""
        return tuple(
            record
            for record in self._index.overlaps(start_frame, end_frame)
            if (tracks is None or record.payload.track in tracks)
            and (effects is None or record.payload.effect in effects)
        )

    def invalidation(
        self,
        start_frame: int,
        end_frame: int,
        *,
        tracks: frozenset[str] | None = None,
    ) -> Invalidation:
        """Return affected edit identities plus canonical frame coverage."""
        records = self.regions(start_frame, end_frame, tracks=tracks)
        coverage = coverage_snapshot(records, start=start_frame, end=end_frame)
        return Invalidation(records, coverage)

    def snapshot(self) -> VideoEditSnapshot:
        """Return an immutable insertion-ordered snapshot."""
        return self._index.snapshot()


def create_catalog() -> VideoEditCatalog:
    """Create an empty video edit region catalog."""
    return VideoEditCatalog()
