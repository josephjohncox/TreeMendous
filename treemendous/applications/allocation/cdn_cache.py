"""CDN object byte-segment residency and request coverage."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from treemendous.applications._shared.allocation import (
    AllocationHandle,
    AllocatorCheckpoint,
    ContiguousAllocator,
    FragmentationDiagnostics,
)
from treemendous.domain import Span, validate_coordinate, validate_length


@dataclass(frozen=True)
class CachedSegment:
    """One exact resident byte segment of a cached object."""

    handle: AllocationHandle
    cache_key: str

    @property
    def byte_range(self) -> Span:
        return self.handle.span


@dataclass(frozen=True)
class RequestCoverage:
    """Resident and missing portions of one byte-range request."""

    requested: Span
    resident_ranges: tuple[Span, ...]
    missing_ranges: tuple[Span, ...]
    covered_bytes: int

    @property
    def fully_resident(self) -> bool:
        return not self.missing_ranges


@dataclass(frozen=True)
class CacheDiagnostics:
    """Residency, eviction, and fragmentation counters."""

    resident_bytes: int
    evictions: int
    fragmentation: FragmentationDiagnostics


@dataclass(frozen=True)
class CacheSnapshot:
    """Immutable segment residency state."""

    object_id: str
    object_size: int
    segments: tuple[CachedSegment, ...]
    missing_ranges: tuple[Span, ...]
    diagnostics: CacheDiagnostics


@dataclass(frozen=True)
class CacheCheckpoint:
    """Restorable cache state."""

    allocator: AllocatorCheckpoint
    segments: tuple[CachedSegment, ...]
    evictions: int


class CDNByteRangeCache:
    """Track non-overlapping resident byte segments for one CDN object."""

    def __init__(self, object_size: int, *, object_id: str = "object") -> None:
        validate_length(object_size)
        if not object_id:
            raise ValueError("object_id must be nonempty")
        self._object_size = object_size
        self._object_id = object_id
        self._allocator = ContiguousAllocator((0, object_size))
        self._segments: dict[int, CachedSegment] = {}
        self._evictions = 0
        self._lock = RLock()

    def cache_segment(
        self, start: int, length: int, *, cache_key: str
    ) -> CachedSegment:
        """Mark one exact, previously absent byte segment resident."""
        validate_coordinate(start, "start")
        validate_length(length)
        if not isinstance(cache_key, str) or not cache_key:
            raise ValueError("cache_key must be a nonempty string")
        with self._lock:
            handle = self._allocator.reserve(
                start, length, owner=self._object_id, idempotency_key=cache_key
            )
            prior = self._segments.get(handle.allocation_id)
            if prior is not None:
                return prior
            segment = CachedSegment(handle, cache_key)
            self._segments[handle.allocation_id] = segment
            return segment

    def evict(self, segment: CachedSegment) -> None:
        """Evict an exact live segment and restore request-miss geometry."""
        if not isinstance(segment, CachedSegment):
            raise TypeError("segment must be a CachedSegment")
        with self._lock:
            if self._segments.get(segment.handle.allocation_id) != segment:
                from treemendous.applications._shared.allocation import (
                    StaleAllocationError,
                )

                raise StaleAllocationError("cached segment is stale or foreign")
            self._allocator.free(segment.handle, owner=self._object_id)
            del self._segments[segment.handle.allocation_id]
            self._evictions += 1

    def request_coverage(self, start: int, length: int) -> RequestCoverage:
        """Classify an in-object request into resident and missing byte ranges."""
        validate_coordinate(start, "start")
        validate_length(length)
        requested = Span(start, start + length)
        if requested.start < 0 or requested.end > self._object_size:
            raise ValueError("request must be contained in the object")
        with self._lock:
            missing = self._intersections(
                requested, self._allocator.snapshot().free_ranges
            )
            resident = self._intersections(
                requested,
                tuple(segment.byte_range for segment in self._segments.values()),
            )
            return RequestCoverage(
                requested=requested,
                resident_ranges=resident,
                missing_ranges=missing,
                covered_bytes=sum(span.length for span in resident),
            )

    def snapshot(self) -> CacheSnapshot:
        """Return all resident segments, misses, and cache diagnostics."""
        with self._lock:
            state = self._allocator.snapshot()
            segments = tuple(
                sorted(self._segments.values(), key=lambda item: item.byte_range)
            )
            return CacheSnapshot(
                object_id=self._object_id,
                object_size=self._object_size,
                segments=segments,
                missing_ranges=state.free_ranges,
                diagnostics=CacheDiagnostics(
                    resident_bytes=state.diagnostics.allocated_space,
                    evictions=self._evictions,
                    fragmentation=state.diagnostics,
                ),
            )

    def checkpoint(self) -> CacheCheckpoint:
        """Capture complete cache residency state."""
        with self._lock:
            return CacheCheckpoint(
                self._allocator.checkpoint(), self.snapshot().segments, self._evictions
            )

    def restore(self, checkpoint: CacheCheckpoint) -> None:
        """Atomically restore a valid local cache checkpoint."""
        if not isinstance(checkpoint, CacheCheckpoint):
            raise TypeError("checkpoint must be a CacheCheckpoint")
        self._allocator.validate_checkpoint_geometry(
            checkpoint.allocator, reserved_ranges=()
        )
        validate_coordinate(checkpoint.evictions, "evictions")
        if checkpoint.evictions < 0:
            raise ValueError("evictions must be nonnegative")
        records = {
            record.handle: record for record in checkpoint.allocator.records
        }
        staged: dict[int, CachedSegment] = {}
        for segment in checkpoint.segments:
            if (
                not isinstance(segment, CachedSegment)
                or segment.handle not in records
                or records[segment.handle].idempotency_key != segment.cache_key
                or segment.handle.owner != self._object_id
                or not isinstance(segment.cache_key, str)
                or not segment.cache_key
                or segment.handle.allocation_id in staged
            ):
                raise ValueError("checkpoint contains invalid cached-segment metadata")
            staged[segment.handle.allocation_id] = segment
        if len(staged) != len(records):
            raise ValueError("checkpoint is missing cached-segment metadata")
        with self._lock:
            self._allocator.restore(checkpoint.allocator)
            self._segments = staged
            self._evictions = checkpoint.evictions

    @staticmethod
    def _intersections(requested: Span, spans: tuple[Span, ...]) -> tuple[Span, ...]:
        intersections = sorted(
            Span(max(requested.start, span.start), min(requested.end, span.end))
            for span in spans
            if span.start < requested.end and requested.start < span.end
        )
        normalized: list[Span] = []
        for span in intersections:
            if normalized and span.start <= normalized[-1].end:
                normalized[-1] = Span(
                    normalized[-1].start, max(normalized[-1].end, span.end)
                )
            else:
                normalized.append(span)
        return tuple(normalized)


def create_application(**kwargs: object) -> CDNByteRangeCache:
    """Registry factory for :class:`CDNByteRangeCache`."""
    return CDNByteRangeCache(**kwargs)  # type: ignore[arg-type]
