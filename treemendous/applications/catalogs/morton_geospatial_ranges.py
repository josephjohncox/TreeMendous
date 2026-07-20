"""Approximate Morton-band geospatial catalog with exact result filtering.

This module does not claim exact Cartesian indexing. Morton intervals generate
a candidate superset; every public spatial result is explicitly filtered using
the retained Cartesian bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from treemendous.applications._shared.interval_records import (
    IntervalRecord,
    IntervalRecordIndex,
    IntervalRecordSnapshot,
    RecordHandle,
)

_DEFAULT_BITS = 16


def _bits(bits: int) -> int:
    if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= 31:
        raise ValueError("bits must be an integer from 1 through 31")
    return bits


def morton_encode(x: int, y: int, *, bits: int = _DEFAULT_BITS) -> int:
    """Interleave unsigned integer coordinates into a Morton code."""
    width = _bits(bits)
    limit = 1 << width
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (x, y)):
        raise TypeError("coordinates must be integers")
    if not (0 <= x < limit and 0 <= y < limit):
        raise ValueError("coordinates are outside the configured Morton domain")
    code = 0
    for bit in range(width):
        code |= ((x >> bit) & 1) << (2 * bit)
        code |= ((y >> bit) & 1) << (2 * bit + 1)
    return code


def morton_decompose(code: int, *, bits: int = _DEFAULT_BITS) -> tuple[int, int]:
    """Deinterleave a Morton code back into ``(x, y)`` integers."""
    width = _bits(bits)
    if isinstance(code, bool) or not isinstance(code, int):
        raise TypeError("code must be an integer")
    if not 0 <= code < (1 << (2 * width)):
        raise ValueError("code is outside the configured Morton domain")
    x = 0
    y = 0
    for bit in range(width):
        x |= ((code >> (2 * bit)) & 1) << bit
        y |= ((code >> (2 * bit + 1)) & 1) << bit
    return x, y


@dataclass(frozen=True)
class GeoRegion:
    """One retained Cartesian rectangle and user label."""

    item_id: str
    min_x: int
    min_y: int
    max_x: int
    max_y: int
    label: str


GeoRecord = IntervalRecord[str, GeoRegion]
GeoHandle = RecordHandle[str]
GeoSnapshot = IntervalRecordSnapshot[str, GeoRegion]


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _rectangle(
    min_x: int, min_y: int, max_x: int, max_y: int, bits: int
) -> tuple[int, int]:
    width = _bits(bits)
    limit = 1 << width
    coordinates = (min_x, min_y, max_x, max_y)
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in coordinates
    ):
        raise TypeError("rectangle coordinates must be integers")
    if not (0 <= min_x < max_x <= limit and 0 <= min_y < max_y <= limit):
        raise ValueError("rectangle must be nonempty and inside the Morton domain")
    return (
        morton_encode(min_x, min_y, bits=width),
        morton_encode(max_x - 1, max_y - 1, bits=width) + 1,
    )


class MortonGeospatialCatalog:
    """Morton candidate-band catalog with mandatory Cartesian filtering."""

    def __init__(self, *, bits: int = _DEFAULT_BITS) -> None:
        self.bits = _bits(bits)
        self._index = IntervalRecordIndex[str, GeoRegion](lambda value: value)

    def add(
        self,
        item_id: str,
        min_x: int,
        min_y: int,
        max_x: int,
        max_y: int,
        *,
        label: str,
    ) -> GeoHandle:
        """Add one half-open Cartesian region as an approximate Morton band."""
        start, end = _rectangle(min_x, min_y, max_x, max_y, self.bits)
        region = GeoRegion(
            _text(item_id, "item_id"),
            min_x,
            min_y,
            max_x,
            max_y,
            _text(label, "label"),
        )
        return self._index.insert(item_id, start, end, region)

    def update(
        self,
        handle: GeoHandle,
        *,
        min_x: int | None = None,
        min_y: int | None = None,
        max_x: int | None = None,
        max_y: int | None = None,
        label: str | None = None,
    ) -> GeoRecord:
        """Update retained Cartesian bounds while preserving identity."""
        current = self._index.get(handle)
        region = replace(
            current.payload,
            min_x=current.payload.min_x if min_x is None else min_x,
            min_y=current.payload.min_y if min_y is None else min_y,
            max_x=current.payload.max_x if max_x is None else max_x,
            max_y=current.payload.max_y if max_y is None else max_y,
            label=current.payload.label if label is None else _text(label, "label"),
        )
        start, end = _rectangle(
            region.min_x, region.min_y, region.max_x, region.max_y, self.bits
        )
        return self._index.update(
            handle, owner=handle.owner, start=start, end=end, payload=region
        )

    def remove(self, handle: GeoHandle) -> GeoRecord:
        """Remove exactly one geospatial identity."""
        return self._index.remove(handle, owner=handle.owner)

    def approximate_candidates(
        self, min_x: int, min_y: int, max_x: int, max_y: int
    ) -> tuple[GeoRecord, ...]:
        """Return the Morton-band candidate superset without exact filtering."""
        start, end = _rectangle(min_x, min_y, max_x, max_y, self.bits)
        return self._index.overlaps(start, end)

    def search(
        self, min_x: int, min_y: int, max_x: int, max_y: int
    ) -> tuple[GeoRecord, ...]:
        """Return exact rectangle intersections after false-positive removal."""
        candidates = self.approximate_candidates(min_x, min_y, max_x, max_y)
        return tuple(
            record
            for record in candidates
            if record.payload.min_x < max_x
            and min_x < record.payload.max_x
            and record.payload.min_y < max_y
            and min_y < record.payload.max_y
        )

    def snapshot(self) -> GeoSnapshot:
        """Return an immutable insertion-ordered snapshot."""
        return self._index.snapshot()


def create_catalog(*, bits: int = _DEFAULT_BITS) -> MortonGeospatialCatalog:
    """Create an empty approximate Morton geospatial catalog."""
    return MortonGeospatialCatalog(bits=bits)
