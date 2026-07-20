"""Contiguous filesystem block extent allocation."""

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
class FileExtent:
    """A contiguous block extent owned by one file identity."""

    handle: AllocationHandle
    file_id: Hashable
    block_size: int

    @property
    def blocks(self) -> Span:
        return self.handle.span

    @property
    def byte_range(self) -> Span:
        return Span(
            self.handle.start * self.block_size, self.handle.end * self.block_size
        )


@dataclass(frozen=True)
class DiskSnapshot:
    """Immutable filesystem allocation state."""

    total_blocks: int
    block_size: int
    metadata_blocks: Span | None
    extents: tuple[FileExtent, ...]
    free_extents: tuple[Span, ...]
    diagnostics: FragmentationDiagnostics


@dataclass(frozen=True)
class DiskCheckpoint:
    """Restorable disk allocation state."""

    allocator: AllocatorCheckpoint
    extents: tuple[FileExtent, ...]


class DiskBlockAllocator:
    """Allocate file extents in whole blocks, reserving leading metadata blocks."""

    def __init__(
        self,
        total_blocks: int,
        *,
        block_size: int = 4096,
        metadata_blocks: int = 1,
        fit_policy: FitPolicy | str = FitPolicy.BEST,
    ) -> None:
        validate_length(total_blocks)
        validate_length(block_size)
        validate_coordinate(metadata_blocks, "metadata_blocks")
        if not 0 <= metadata_blocks < total_blocks:
            raise ValueError("metadata_blocks must be within the disk")
        try:
            self._policy = FitPolicy(fit_policy)
        except (TypeError, ValueError) as error:
            raise ValueError("fit policy must be first, best, or worst") from error
        reserved = (Span(0, metadata_blocks),) if metadata_blocks else ()
        self._allocator = ContiguousAllocator((0, total_blocks), reserved=reserved)
        self._reserved = reserved
        self._block_size = block_size
        self._metadata = reserved[0] if reserved else None
        self._extents: dict[int, FileExtent] = {}
        self._lock = RLock()

    def allocate_extent(
        self,
        file_id: Hashable,
        block_count: int,
        *,
        policy: FitPolicy | str | None = None,
    ) -> FileExtent:
        """Allocate one contiguous file extent in whole blocks."""
        validate_length(block_count)
        selected = self._policy if policy is None else FitPolicy(policy)
        with self._lock:
            handle = self._allocator.allocate(
                block_count, owner=file_id, policy=selected
            )
            extent = FileExtent(handle, file_id, self._block_size)
            self._extents[handle.allocation_id] = extent
            return extent

    def free_extent(self, extent: FileExtent, *, file_id: Hashable) -> None:
        """Return an exact live file extent to free space."""
        if not isinstance(extent, FileExtent):
            raise TypeError("extent must be a FileExtent")
        with self._lock:
            if self._extents.get(extent.handle.allocation_id) != extent:
                from treemendous.applications._shared.allocation import (
                    StaleAllocationError,
                )

                raise StaleAllocationError("file extent is stale or foreign")
            self._allocator.free(extent.handle, owner=file_id)
            del self._extents[extent.handle.allocation_id]

    def extents_for(self, file_id: Hashable) -> tuple[FileExtent, ...]:
        """Return a file's extents in physical block order."""
        with self._lock:
            return tuple(
                sorted(
                    (
                        item
                        for item in self._extents.values()
                        if item.file_id == file_id
                    ),
                    key=lambda item: item.blocks,
                )
            )

    def snapshot(self) -> DiskSnapshot:
        """Return extents, free geometry, and fragmentation diagnostics."""
        with self._lock:
            state = self._allocator.snapshot()
            return DiskSnapshot(
                total_blocks=state.domain.measure,
                block_size=self._block_size,
                metadata_blocks=self._metadata,
                extents=tuple(
                    sorted(self._extents.values(), key=lambda item: item.blocks)
                ),
                free_extents=state.free_ranges,
                diagnostics=state.diagnostics,
            )

    def checkpoint(self) -> DiskCheckpoint:
        """Capture complete disk state."""
        with self._lock:
            return DiskCheckpoint(self._allocator.checkpoint(), self.snapshot().extents)

    def restore(self, checkpoint: DiskCheckpoint) -> None:
        """Atomically restore a checkpoint created by this disk allocator."""
        if not isinstance(checkpoint, DiskCheckpoint):
            raise TypeError("checkpoint must be a DiskCheckpoint")
        self._allocator.validate_checkpoint_geometry(
            checkpoint.allocator, reserved_ranges=self._reserved
        )
        allocator_handles = {record.handle for record in checkpoint.allocator.records}
        staged: dict[int, FileExtent] = {}
        for extent in checkpoint.extents:
            if (
                not isinstance(extent, FileExtent)
                or extent.handle not in allocator_handles
                or extent.file_id != extent.handle.owner
                or extent.block_size != self._block_size
                or extent.handle.allocation_id in staged
            ):
                raise ValueError("checkpoint contains invalid file extent metadata")
            staged[extent.handle.allocation_id] = extent
        if len(staged) != len(allocator_handles):
            raise ValueError("checkpoint is missing file extent metadata")
        with self._lock:
            self._allocator.restore(checkpoint.allocator)
            self._extents = staged


def create_application(**kwargs: object) -> DiskBlockAllocator:
    """Registry factory for :class:`DiskBlockAllocator`."""
    return DiskBlockAllocator(**kwargs)  # type: ignore[arg-type]
