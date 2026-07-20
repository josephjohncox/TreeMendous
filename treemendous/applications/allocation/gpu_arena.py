"""GPU device-memory arena with stream-ordered deferred reclamation."""

from __future__ import annotations

from collections.abc import Hashable
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
class GPUBuffer:
    """One aligned device-memory allocation owned by a stream."""

    handle: AllocationHandle
    stream: Hashable
    alignment: int

    @property
    def device_range(self) -> Span:
        return self.handle.span


@dataclass(frozen=True)
class DeferredFree:
    """A buffer retained until its stream reaches a completion epoch."""

    buffer: GPUBuffer
    completion_epoch: int


@dataclass(frozen=True)
class GPUArenaSnapshot:
    """Immutable live/deferred arena state and diagnostics."""

    live_buffers: tuple[GPUBuffer, ...]
    deferred_frees: tuple[DeferredFree, ...]
    completed_epochs: tuple[tuple[Hashable, int], ...]
    free_ranges: tuple[Span, ...]
    diagnostics: FragmentationDiagnostics


@dataclass(frozen=True)
class GPUArenaCheckpoint:
    """Restorable device arena state."""

    allocator: AllocatorCheckpoint
    buffers: tuple[GPUBuffer, ...]
    deferred_frees: tuple[DeferredFree, ...]
    completed_epochs: tuple[tuple[Hashable, int], ...]


class GPUMemoryArena:
    """Allocate aligned buffers and reclaim them only after stream completion."""

    def __init__(
        self,
        capacity: int,
        *,
        base_address: int = 0,
        device_alignment: int = 256,
    ) -> None:
        validate_length(capacity)
        validate_coordinate(base_address, "base_address")
        self._validate_alignment(device_alignment)
        if base_address % device_alignment:
            raise ValueError("base_address must satisfy device alignment")
        self._alignment = device_alignment
        self._allocator = ContiguousAllocator((base_address, base_address + capacity))
        self._buffers: dict[int, GPUBuffer] = {}
        self._deferred: dict[int, DeferredFree] = {}
        self._completed: dict[Hashable, int] = {}
        self._lock = RLock()

    def allocate(
        self,
        size: int,
        *,
        stream: Hashable,
        alignment: int | None = None,
    ) -> GPUBuffer:
        """Allocate one device buffer owned by a stream."""
        validate_length(size)
        selected = self._alignment if alignment is None else alignment
        self._validate_alignment(selected)
        if selected < self._alignment:
            raise ValueError("alignment cannot be less than device alignment")
        with self._lock:
            handle = self._allocator.allocate(size, owner=stream, alignment=selected)
            buffer = GPUBuffer(handle, stream, selected)
            self._buffers[handle.allocation_id] = buffer
            return buffer

    def defer_free(
        self,
        buffer: GPUBuffer,
        *,
        stream: Hashable,
        completion_epoch: int,
    ) -> DeferredFree:
        """Queue a buffer for reclamation on its owning stream."""
        if not isinstance(buffer, GPUBuffer):
            raise TypeError("buffer must be a GPUBuffer")
        validate_coordinate(completion_epoch, "completion_epoch")
        if completion_epoch < 0:
            raise ValueError("completion_epoch must be nonnegative")
        with self._lock:
            self._require_live(buffer)
            if stream != buffer.stream:
                from treemendous.applications._shared.allocation import (
                    ForeignAllocationError,
                )

                raise ForeignAllocationError("buffer belongs to another stream")
            if buffer.handle.allocation_id in self._deferred:
                raise ValueError("buffer already has a deferred free")
            if completion_epoch <= self._completed.get(stream, -1):
                self._allocator.free(buffer.handle, owner=stream)
                del self._buffers[buffer.handle.allocation_id]
                return DeferredFree(buffer, completion_epoch)
            deferred = DeferredFree(buffer, completion_epoch)
            self._deferred[buffer.handle.allocation_id] = deferred
            return deferred

    def advance_completion(
        self, stream: Hashable, completion_epoch: int
    ) -> tuple[GPUBuffer, ...]:
        """Advance one stream monotonically and reclaim eligible buffers."""
        validate_coordinate(completion_epoch, "completion_epoch")
        if completion_epoch < 0:
            raise ValueError("completion_epoch must be nonnegative")
        with self._lock:
            previous = self._completed.get(stream, -1)
            if completion_epoch < previous:
                raise ValueError("stream completion epoch cannot move backwards")
            eligible = tuple(
                item.buffer
                for item in sorted(
                    self._deferred.values(),
                    key=lambda item: (
                        item.completion_epoch,
                        item.buffer.handle.allocation_id,
                    ),
                )
                if item.buffer.stream == stream
                and item.completion_epoch <= completion_epoch
            )
            checkpoint = self._allocator.checkpoint()
            old_buffers = dict(self._buffers)
            old_deferred = dict(self._deferred)
            try:
                for buffer in eligible:
                    self._allocator.free(buffer.handle, owner=stream)
                    del self._buffers[buffer.handle.allocation_id]
                    del self._deferred[buffer.handle.allocation_id]
            except Exception:
                self._allocator.restore(checkpoint)
                self._buffers = old_buffers
                self._deferred = old_deferred
                raise
            self._completed[stream] = completion_epoch
            return eligible

    def snapshot(self) -> GPUArenaSnapshot:
        """Return live/deferred buffers, stream epochs, and free-space metrics."""
        with self._lock:
            state = self._allocator.snapshot()
            deferred_ids = set(self._deferred)
            return GPUArenaSnapshot(
                live_buffers=tuple(
                    sorted(
                        (
                            item
                            for key, item in self._buffers.items()
                            if key not in deferred_ids
                        ),
                        key=lambda item: item.device_range,
                    )
                ),
                deferred_frees=tuple(
                    sorted(
                        self._deferred.values(),
                        key=lambda item: item.buffer.device_range,
                    )
                ),
                completed_epochs=tuple(self._completed.items()),
                free_ranges=state.free_ranges,
                diagnostics=state.diagnostics,
            )

    def checkpoint(self) -> GPUArenaCheckpoint:
        """Capture complete stream and allocator state."""
        with self._lock:
            return GPUArenaCheckpoint(
                self._allocator.checkpoint(),
                tuple(
                    sorted(self._buffers.values(), key=lambda item: item.device_range)
                ),
                self.snapshot().deferred_frees,
                tuple(self._completed.items()),
            )

    def restore(self, checkpoint: GPUArenaCheckpoint) -> None:
        """Atomically restore a valid local arena checkpoint."""
        if not isinstance(checkpoint, GPUArenaCheckpoint):
            raise TypeError("checkpoint must be a GPUArenaCheckpoint")
        self._allocator.validate_checkpoint_geometry(
            checkpoint.allocator, reserved_ranges=()
        )
        handles = {record.handle for record in checkpoint.allocator.records}
        buffers: dict[int, GPUBuffer] = {}
        for buffer in checkpoint.buffers:
            if not isinstance(buffer, GPUBuffer):
                raise TypeError("checkpoint buffers must be GPUBuffer values")
            self._validate_alignment(buffer.alignment)
            if (
                buffer.handle not in handles
                or buffer.stream != buffer.handle.owner
                or buffer.alignment < self._alignment
                or buffer.handle.start % buffer.alignment
                or buffer.handle.allocation_id in buffers
            ):
                raise ValueError("checkpoint contains invalid GPU buffer metadata")
            buffers[buffer.handle.allocation_id] = buffer
        if len(buffers) != len(handles):
            raise ValueError("checkpoint is missing GPU buffer metadata")
        deferred: dict[int, DeferredFree] = {}
        for item in checkpoint.deferred_frees:
            if not isinstance(item, DeferredFree) or not isinstance(
                item.buffer, GPUBuffer
            ):
                raise TypeError("checkpoint deferred frees must contain GPU buffers")
            allocation_id = item.buffer.handle.allocation_id
            if (
                buffers.get(allocation_id) != item.buffer
                or item.completion_epoch < 0
                or allocation_id in deferred
            ):
                raise ValueError("checkpoint contains invalid deferred-free metadata")
            deferred[allocation_id] = item
        completed: dict[Hashable, int] = {}
        for stream, epoch in checkpoint.completed_epochs:
            validate_coordinate(epoch, "completion_epoch")
            if epoch < 0 or stream in completed:
                raise ValueError("checkpoint contains invalid stream epochs")
            completed[stream] = epoch
        if any(
            item.completion_epoch <= completed.get(item.buffer.stream, -1)
            for item in deferred.values()
        ):
            raise ValueError("checkpoint contains an already-completed deferred free")
        with self._lock:
            self._allocator.restore(checkpoint.allocator)
            self._buffers = buffers
            self._deferred = deferred
            self._completed = completed

    def _require_live(self, buffer: GPUBuffer) -> None:
        if self._buffers.get(buffer.handle.allocation_id) != buffer:
            from treemendous.applications._shared.allocation import StaleAllocationError

            raise StaleAllocationError("GPU buffer is stale or foreign")

    @staticmethod
    def _validate_alignment(value: int) -> None:
        validate_coordinate(value, "alignment")
        if value <= 0 or value & (value - 1):
            raise ValueError("alignment must be a positive power of two")


def create_application(**kwargs: object) -> GPUMemoryArena:
    """Registry factory for :class:`GPUMemoryArena`."""
    return GPUMemoryArena(**kwargs)  # type: ignore[arg-type]
