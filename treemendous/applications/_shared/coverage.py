"""Canonical coverage projection for identity-preserving interval records."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic

from treemendous.applications._shared.interval_records import (
    IntervalRecord,
    OwnerT,
    PayloadT,
    RecordHandle,
)
from treemendous.domain import Span


@dataclass(frozen=True)
class CoverageSegment(Generic[OwnerT]):
    """A maximal half-open segment having one exact active-record set."""

    start: int
    end: int
    record_ids: frozenset[RecordHandle[OwnerT]]

    def __post_init__(self) -> None:
        Span(self.start, self.end)
        if not self.record_ids:
            raise ValueError("coverage segment must contain at least one record")

    @property
    def count(self) -> int:
        """Return the number of distinct covering records."""
        return len(self.record_ids)

    @property
    def handles(self) -> frozenset[RecordHandle[OwnerT]]:
        """Alias exposing the record identities as handles."""
        return self.record_ids

    @property
    def span(self) -> Span:
        """Return this segment as a validated span."""
        return Span(self.start, self.end)


@dataclass(frozen=True)
class CoverageSnapshot(Generic[OwnerT]):
    """Canonical positive-coverage segments and aggregate diagnostics."""

    segments: tuple[CoverageSegment[OwnerT], ...]
    covered_length: int
    maximum_count: int

    def __post_init__(self) -> None:
        actual_length = sum(segment.end - segment.start for segment in self.segments)
        if self.covered_length != actual_length:
            raise ValueError("covered_length does not match segments")
        actual_maximum = max((segment.count for segment in self.segments), default=0)
        if self.maximum_count != actual_maximum:
            raise ValueError("maximum_count does not match segments")


def coverage_segments(
    records: Iterable[IntervalRecord[OwnerT, PayloadT]],
    *,
    start: int | None = None,
    end: int | None = None,
) -> tuple[CoverageSegment[OwnerT], ...]:
    """Project records into maximal positive-coverage segments.

    Supplying ``start`` and ``end`` clips the projection to that half-open
    query span.  They must be supplied together.  Duplicate record identities
    are rejected because a count is explicitly a count of record IDs.
    """
    if (start is None) != (end is None):
        raise ValueError("start and end must be supplied together")
    bounds = None if start is None else Span(start, end)  # type: ignore[arg-type]

    starts: dict[int, set[RecordHandle[OwnerT]]] = {}
    ends: dict[int, set[RecordHandle[OwnerT]]] = {}
    seen: set[RecordHandle[OwnerT]] = set()
    for record in records:
        if record.handle in seen:
            raise ValueError(f"duplicate record identity: {record.handle!r}")
        seen.add(record.handle)
        left = record.start
        right = record.end
        if bounds is not None:
            left = max(left, bounds.start)
            right = min(right, bounds.end)
            if left >= right:
                continue
        starts.setdefault(left, set()).add(record.handle)
        ends.setdefault(right, set()).add(record.handle)

    points = sorted(starts.keys() | ends.keys())
    if len(points) < 2:
        return ()

    active: set[RecordHandle[OwnerT]] = set()
    result: list[CoverageSegment[OwnerT]] = []
    for index, point in enumerate(points[:-1]):
        # End events precede start events at an equal half-open boundary.
        active.difference_update(ends.get(point, ()))
        active.update(starts.get(point, ()))
        next_point = points[index + 1]
        if not active or point == next_point:
            continue
        identities = frozenset(active)
        if result and result[-1].end == point and result[-1].record_ids == identities:
            previous = result[-1]
            result[-1] = CoverageSegment(previous.start, next_point, identities)
        else:
            result.append(CoverageSegment(point, next_point, identities))
    return tuple(result)


def coverage_snapshot(
    records: Iterable[IntervalRecord[OwnerT, PayloadT]],
    *,
    start: int | None = None,
    end: int | None = None,
) -> CoverageSnapshot[OwnerT]:
    """Return canonical coverage plus covered-length and peak-count metrics."""
    segments = coverage_segments(records, start=start, end=end)
    return CoverageSnapshot(
        segments=segments,
        covered_length=sum(segment.end - segment.start for segment in segments),
        maximum_count=max((segment.count for segment in segments), default=0),
    )
