"""Database tablespace page extent allocation and freed-page reuse."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from threading import RLock

from treemendous.applications._shared.allocation import (
    AllocationHandle,
    AllocatorCheckpoint,
    ContiguousAllocator,
    FitPolicy,
    FragmentationDiagnostics,
)
from treemendous.domain import Span, validate_coordinate, validate_length


@dataclass(frozen=True)
class PageExtent:
    """A contiguous set of pages assigned to one table."""

    handle: AllocationHandle
    table_id: Hashable
    tablespace: str
    reused: bool

    @property
    def pages(self) -> Span:
        return self.handle.span


@dataclass(frozen=True)
class DatabaseDiagnostics:
    """Free-space and reuse counters for a tablespace."""

    fragmentation: FragmentationDiagnostics
    reuse_allocations: int
    pages_ever_freed: int


@dataclass(frozen=True)
class DatabaseSnapshot:
    """Immutable tablespace allocation state."""

    tablespace: str
    page_size: int
    extents: tuple[PageExtent, ...]
    free_pages: tuple[Span, ...]
    diagnostics: DatabaseDiagnostics


@dataclass(frozen=True)
class DatabaseCheckpoint:
    """Restorable database page-allocation state."""

    allocator: AllocatorCheckpoint
    extents: tuple[PageExtent, ...]
    freed_history: tuple[Span, ...]
    reuse_allocations: int


class DatabasePageAllocator:
    """Allocate first-fit table extents and expose deterministic page reuse."""

    def __init__(
        self,
        total_pages: int,
        *,
        tablespace: str = "main",
        page_size: int = 8192,
        metadata_pages: int = 1,
    ) -> None:
        validate_length(total_pages)
        validate_length(page_size)
        validate_coordinate(metadata_pages, "metadata_pages")
        if not tablespace:
            raise ValueError("tablespace must be nonempty")
        if not 0 <= metadata_pages < total_pages:
            raise ValueError("metadata_pages must be within the tablespace")
        reserved = (Span(0, metadata_pages),) if metadata_pages else ()
        self._allocator = ContiguousAllocator((0, total_pages), reserved=reserved)
        self._reserved = reserved
        self._tablespace = tablespace
        self._page_size = page_size
        self._extents: dict[int, PageExtent] = {}
        self._freed_history: tuple[Span, ...] = ()
        self._reuse_allocations = 0
        self._lock = RLock()

    def allocate_pages(self, table_id: Hashable, page_count: int) -> PageExtent:
        """Allocate a contiguous first-fit page extent for a table."""
        validate_length(page_count)
        with self._lock:
            handle = self._allocator.allocate(
                page_count, owner=table_id, policy=FitPolicy.FIRST
            )
            reused = any(
                span.start < handle.end and handle.start < span.end
                for span in self._freed_history
            )
            extent = PageExtent(handle, table_id, self._tablespace, reused)
            self._extents[handle.allocation_id] = extent
            if reused:
                self._reuse_allocations += 1
            return extent

    def free_pages(self, extent: PageExtent, *, table_id: Hashable) -> None:
        """Free an exact live extent and record it as reusable page space."""
        if not isinstance(extent, PageExtent):
            raise TypeError("extent must be a PageExtent")
        with self._lock:
            if self._extents.get(extent.handle.allocation_id) != extent:
                from treemendous.applications._shared.allocation import (
                    StaleAllocationError,
                )

                raise StaleAllocationError("database extent is stale or foreign")
            self._allocator.free(extent.handle, owner=table_id)
            del self._extents[extent.handle.allocation_id]
            self._freed_history = self._merge((*self._freed_history, extent.pages))

    def snapshot(self) -> DatabaseSnapshot:
        """Return tablespace geometry and reuse diagnostics."""
        with self._lock:
            state = self._allocator.snapshot()
            return DatabaseSnapshot(
                tablespace=self._tablespace,
                page_size=self._page_size,
                extents=tuple(
                    sorted(self._extents.values(), key=lambda item: item.pages)
                ),
                free_pages=state.free_ranges,
                diagnostics=DatabaseDiagnostics(
                    fragmentation=state.diagnostics,
                    reuse_allocations=self._reuse_allocations,
                    pages_ever_freed=sum(span.length for span in self._freed_history),
                ),
            )

    def checkpoint(self) -> DatabaseCheckpoint:
        """Capture complete tablespace state."""
        with self._lock:
            return DatabaseCheckpoint(
                self._allocator.checkpoint(),
                self.snapshot().extents,
                self._freed_history,
                self._reuse_allocations,
            )

    def restore(self, checkpoint: DatabaseCheckpoint) -> None:
        """Atomically restore a valid local checkpoint."""
        if not isinstance(checkpoint, DatabaseCheckpoint):
            raise TypeError("checkpoint must be a DatabaseCheckpoint")
        self._allocator.validate_checkpoint_geometry(
            checkpoint.allocator, reserved_ranges=self._reserved
        )
        validate_coordinate(checkpoint.reuse_allocations, "reuse_allocations")
        if checkpoint.reuse_allocations < 0:
            raise ValueError("reuse_allocations must be nonnegative")
        if any(not isinstance(span, Span) for span in checkpoint.freed_history):
            raise TypeError("freed-page history must contain only Span values")
        if any(
            not self._allocator.domain.contains(span)
            or any(
                reserved.start < span.end and span.start < reserved.end
                for reserved in self._reserved
            )
            for span in checkpoint.freed_history
        ):
            raise ValueError(
                "freed-page history must stay within allocatable page geometry"
            )
        history = self._merge(checkpoint.freed_history)
        if history != checkpoint.freed_history:
            raise ValueError("freed-page history must be normalized")
        handles = {record.handle for record in checkpoint.allocator.records}
        staged: dict[int, PageExtent] = {}
        for extent in checkpoint.extents:
            expected_reused = any(
                span.start < extent.handle.end and extent.handle.start < span.end
                for span in history
            ) if isinstance(extent, PageExtent) else False
            if (
                not isinstance(extent, PageExtent)
                or extent.handle not in handles
                or extent.table_id != extent.handle.owner
                or extent.tablespace != self._tablespace
                or extent.reused != expected_reused
                or extent.handle.allocation_id in staged
            ):
                raise ValueError("checkpoint contains invalid page extent metadata")
            staged[extent.handle.allocation_id] = extent
        if len(staged) != len(handles):
            raise ValueError("checkpoint is missing page extent metadata")
        if checkpoint.reuse_allocations < sum(
            extent.reused for extent in staged.values()
        ):
            raise ValueError(
                "reuse_allocations cannot be less than live reused extents"
            )
        with self._lock:
            self._allocator.restore(checkpoint.allocator)
            self._extents = staged
            self._freed_history = history
            self._reuse_allocations = checkpoint.reuse_allocations

    @staticmethod
    def _merge(spans: tuple[Span, ...]) -> tuple[Span, ...]:
        merged: list[Span] = []
        for span in sorted(spans):
            if merged and span.start <= merged[-1].end:
                merged[-1] = Span(merged[-1].start, max(merged[-1].end, span.end))
            else:
                merged.append(span)
        return tuple(merged)


def create_application(**kwargs: object) -> DatabasePageAllocator:
    """Registry factory for :class:`DatabasePageAllocator`."""
    return DatabasePageAllocator(**kwargs)  # type: ignore[arg-type]
