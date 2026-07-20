"""Object-store multipart byte-range completion tracking."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from treemendous.applications._shared.allocation import (
    AllocationHandle,
    AllocatorCheckpoint,
    ContiguousAllocator,
)
from treemendous.domain import Span, validate_coordinate, validate_length


class PartConflictError(ValueError):
    """Raised when a completed part is replaced without retry intent."""


@dataclass(frozen=True)
class CompletedPart:
    """Identity and byte geometry for one completed upload part."""

    part_number: int
    etag: str
    byte_range: Span
    attempt: int
    handle: AllocationHandle


@dataclass(frozen=True)
class MultipartDiagnostics:
    """Completion counters for an upload."""

    total_parts: int
    completed_parts: int
    missing_bytes: int
    retry_count: int


@dataclass(frozen=True)
class MultipartSnapshot:
    """Immutable completed, missing, and contiguous upload state."""

    upload_id: str
    object_size: int
    part_size: int
    completed: tuple[CompletedPart, ...]
    missing_ranges: tuple[Span, ...]
    contiguous_completion: Span | None
    diagnostics: MultipartDiagnostics


@dataclass(frozen=True)
class MultipartCheckpoint:
    """Restorable multipart completion state."""

    allocator: AllocatorCheckpoint
    completed: tuple[CompletedPart, ...]


class MultipartUploadTracker:
    """Track fixed-grid byte parts with ETag-aware retries.

    A part number determines exactly one byte range.  Replaying the same ETag
    is idempotent; replacing it requires ``retry=True`` and increments the
    attempt counter without changing completion geometry.
    """

    def __init__(self, object_size: int, part_size: int, *, upload_id: str = "upload"):
        validate_length(object_size)
        validate_length(part_size)
        if not upload_id:
            raise ValueError("upload_id must be nonempty")
        self._object_size = object_size
        self._part_size = part_size
        self._upload_id = upload_id
        self._total_parts = (object_size + part_size - 1) // part_size
        self._allocator = ContiguousAllocator((0, object_size))
        self._completed: dict[int, CompletedPart] = {}
        self._lock = RLock()

    def complete_part(
        self,
        part_number: int,
        etag: str,
        *,
        size: int | None = None,
        retry: bool = False,
    ) -> CompletedPart:
        """Record a completed part or an explicitly identified retry."""
        span = self.part_range(part_number)
        if not isinstance(etag, str) or not etag:
            raise ValueError("etag must be a nonempty string")
        if size is not None:
            validate_length(size)
            if size != span.length:
                raise ValueError("part size does not match its expected byte range")
        with self._lock:
            prior = self._completed.get(part_number)
            if prior is not None:
                if prior.etag == etag:
                    return prior
                if not retry:
                    raise PartConflictError(
                        "completed part has a different ETag; retry intent is required"
                    )
                replacement = CompletedPart(
                    part_number, etag, span, prior.attempt + 1, prior.handle
                )
                self._completed[part_number] = replacement
                return replacement
            if retry:
                raise PartConflictError("cannot retry a part that has not completed")
            handle = self._allocator.reserve(
                span.start, span.length, owner=self._upload_id
            )
            completed = CompletedPart(part_number, etag, span, 1, handle)
            self._completed[part_number] = completed
            return completed

    def part_range(self, part_number: int) -> Span:
        """Return the canonical byte range for a one-based part number."""
        validate_coordinate(part_number, "part_number")
        if not 1 <= part_number <= self._total_parts:
            raise ValueError("part_number is outside this upload")
        start = (part_number - 1) * self._part_size
        return Span(start, min(start + self._part_size, self._object_size))

    def snapshot(self) -> MultipartSnapshot:
        """Return completed parts, missing bytes, prefix completion, and counters."""
        with self._lock:
            completed = tuple(
                self._completed[index] for index in sorted(self._completed)
            )
            missing = self._allocator.snapshot().free_ranges
            cursor = 0
            for part in completed:
                if part.byte_range.start != cursor:
                    break
                cursor = part.byte_range.end
            contiguous = Span(0, cursor) if cursor else None
            return MultipartSnapshot(
                upload_id=self._upload_id,
                object_size=self._object_size,
                part_size=self._part_size,
                completed=completed,
                missing_ranges=missing,
                contiguous_completion=contiguous,
                diagnostics=MultipartDiagnostics(
                    total_parts=self._total_parts,
                    completed_parts=len(completed),
                    missing_bytes=sum(span.length for span in missing),
                    retry_count=sum(part.attempt - 1 for part in completed),
                ),
            )

    def checkpoint(self) -> MultipartCheckpoint:
        """Capture complete upload completion state."""
        with self._lock:
            return MultipartCheckpoint(
                self._allocator.checkpoint(), self.snapshot().completed
            )

    def restore(self, checkpoint: MultipartCheckpoint) -> None:
        """Atomically restore a valid checkpoint from this upload tracker."""
        if not isinstance(checkpoint, MultipartCheckpoint):
            raise TypeError("checkpoint must be a MultipartCheckpoint")
        self._allocator.validate_checkpoint_geometry(
            checkpoint.allocator, reserved_ranges=()
        )
        handles = {record.handle for record in checkpoint.allocator.records}
        staged: dict[int, CompletedPart] = {}
        for part in checkpoint.completed:
            if not isinstance(part, CompletedPart):
                raise TypeError("checkpoint completed entries must be CompletedPart")
            expected = self.part_range(part.part_number)
            if (
                part.handle not in handles
                or part.handle.owner != self._upload_id
                or part.byte_range != expected
                or part.handle.span != expected
                or not isinstance(part.etag, str)
                or not part.etag
                or part.attempt <= 0
                or part.part_number in staged
            ):
                raise ValueError("checkpoint contains invalid completed-part metadata")
            staged[part.part_number] = part
        if len(staged) != len(handles):
            raise ValueError("checkpoint is missing completed-part metadata")
        with self._lock:
            self._allocator.restore(checkpoint.allocator)
            self._completed = staged


def create_application(**kwargs: object) -> MultipartUploadTracker:
    """Registry factory for :class:`MultipartUploadTracker`."""
    return MultipartUploadTracker(**kwargs)  # type: ignore[arg-type]
