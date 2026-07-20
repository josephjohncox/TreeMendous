"""Heap free-space engine with payload alignment and allocator overhead."""

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
class HeapBlock:
    """One live heap allocation, including its inaccessible allocator overhead."""

    raw_handle: AllocationHandle
    payload: Span
    requested_size: int
    header_size: int
    redzone_size: int
    alignment: int

    @property
    def owner(self) -> Hashable:
        return self.raw_handle.owner


@dataclass(frozen=True)
class HeapSnapshot:
    """Immutable heap state and fragmentation diagnostics."""

    capacity: int
    blocks: tuple[HeapBlock, ...]
    free_ranges: tuple[Span, ...]
    diagnostics: FragmentationDiagnostics


@dataclass(frozen=True)
class HeapCheckpoint:
    """Restorable heap geometry and block metadata."""

    allocator: AllocatorCheckpoint
    blocks: tuple[HeapBlock, ...]


class HeapAllocator:
    """Allocate aligned payloads with headers and symmetric redzones.

    Fit policy ranks complete free chunks.  Payload alignment, rather than raw
    block alignment, is measured from address zero.  A failed request leaves
    both allocator geometry and block metadata unchanged.
    """

    def __init__(
        self,
        capacity: int,
        *,
        base_address: int = 0,
        header_size: int = 8,
        redzone_size: int = 0,
        default_alignment: int = 8,
        fit_policy: FitPolicy | str = FitPolicy.FIRST,
    ) -> None:
        validate_length(capacity)
        validate_coordinate(base_address, "base_address")
        self._validate_nonnegative(header_size, "header_size")
        self._validate_nonnegative(redzone_size, "redzone_size")
        self._validate_alignment(default_alignment)
        self._policy = self._coerce_policy(fit_policy)
        self._header_size = header_size
        self._redzone_size = redzone_size
        self._default_alignment = default_alignment
        self._allocator = ContiguousAllocator((base_address, base_address + capacity))
        self._blocks: dict[int, HeapBlock] = {}
        self._lock = RLock()

    def allocate(
        self,
        size: int,
        *,
        owner: Hashable,
        alignment: int | None = None,
        policy: FitPolicy | str | None = None,
    ) -> HeapBlock:
        """Allocate one payload and return its complete layout."""
        validate_length(size)
        selected_alignment = self._default_alignment if alignment is None else alignment
        self._validate_alignment(selected_alignment)
        selected_policy = self._policy if policy is None else self._coerce_policy(policy)
        prefix = self._header_size + self._redzone_size
        suffix = self._redzone_size
        total = prefix + size + suffix
        with self._lock:
            candidates: list[tuple[int, int, int]] = []
            for chunk in self._allocator.snapshot().free_ranges:
                payload_start = self._align_up(chunk.start + prefix, selected_alignment)
                raw_start = payload_start - prefix
                if raw_start + total <= chunk.end:
                    candidates.append((chunk.length, raw_start, payload_start))
            if not candidates:
                from treemendous.applications._shared.allocation import (
                    AllocationUnavailableError,
                )

                raise AllocationUnavailableError(
                    "no free heap chunk satisfies payload size and alignment"
                )
            if selected_policy is FitPolicy.FIRST:
                _, raw_start, payload_start = min(candidates, key=lambda item: item[1])
            elif selected_policy is FitPolicy.BEST:
                _, raw_start, payload_start = min(
                    candidates, key=lambda item: (item[0], item[1])
                )
            else:
                _, raw_start, payload_start = min(
                    candidates, key=lambda item: (-item[0], item[1])
                )
            raw = self._allocator.reserve(raw_start, total, owner=owner)
            block = HeapBlock(
                raw_handle=raw,
                payload=Span(payload_start, payload_start + size),
                requested_size=size,
                header_size=self._header_size,
                redzone_size=self._redzone_size,
                alignment=selected_alignment,
            )
            self._blocks[raw.allocation_id] = block
            return block

    def free(self, block: HeapBlock, *, owner: Hashable) -> None:
        """Release a live block after validating heap and owner identity."""
        if not isinstance(block, HeapBlock):
            raise TypeError("block must be a HeapBlock")
        with self._lock:
            if self._blocks.get(block.raw_handle.allocation_id) != block:
                from treemendous.applications._shared.allocation import (
                    StaleAllocationError,
                )

                raise StaleAllocationError("heap block is stale or foreign")
            self._allocator.free(block.raw_handle, owner=owner)
            del self._blocks[block.raw_handle.allocation_id]

    def snapshot(self) -> HeapSnapshot:
        """Return ordered layouts, free ranges, and fragmentation metrics."""
        with self._lock:
            state = self._allocator.snapshot()
            return HeapSnapshot(
                capacity=state.domain.measure,
                blocks=tuple(sorted(self._blocks.values(), key=lambda item: item.payload)),
                free_ranges=state.free_ranges,
                diagnostics=state.diagnostics,
            )

    def checkpoint(self) -> HeapCheckpoint:
        """Capture complete restorable heap state."""
        with self._lock:
            return HeapCheckpoint(self._allocator.checkpoint(), self.snapshot().blocks)

    def restore(self, checkpoint: HeapCheckpoint) -> None:
        """Atomically restore a structurally valid checkpoint from this heap."""
        if not isinstance(checkpoint, HeapCheckpoint):
            raise TypeError("checkpoint must be a HeapCheckpoint")
        self._allocator.validate_checkpoint_geometry(
            checkpoint.allocator, reserved_ranges=()
        )
        staged: dict[int, HeapBlock] = {}
        allocator_handles = {record.handle for record in checkpoint.allocator.records}
        for block in checkpoint.blocks:
            if not isinstance(block, HeapBlock) or block.raw_handle not in allocator_handles:
                raise ValueError("checkpoint block metadata does not match allocator")
            raw = block.raw_handle.span
            expected_start = raw.start + block.header_size + block.redzone_size
            expected_end = raw.end - block.redzone_size
            self._validate_alignment(block.alignment)
            if (
                block.header_size != self._header_size
                or block.redzone_size != self._redzone_size
                or block.payload != Span(expected_start, expected_end)
                or block.requested_size <= 0
                or block.requested_size != block.payload.length
                or block.payload.start % block.alignment
                or block.raw_handle.allocation_id in staged
            ):
                raise ValueError("checkpoint contains an invalid heap layout")
            staged[block.raw_handle.allocation_id] = block
        if len(staged) != len(allocator_handles):
            raise ValueError("checkpoint is missing heap block metadata")
        with self._lock:
            self._allocator.restore(checkpoint.allocator)
            self._blocks = staged

    @staticmethod
    def _align_up(value: int, alignment: int) -> int:
        return -((-value) // alignment) * alignment

    @staticmethod
    def _validate_nonnegative(value: int, name: str) -> None:
        validate_coordinate(value, name)
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")

    @staticmethod
    def _validate_alignment(value: int) -> None:
        validate_coordinate(value, "alignment")
        if value <= 0 or value & (value - 1):
            raise ValueError("alignment must be a positive power of two")

    @staticmethod
    def _coerce_policy(value: FitPolicy | str) -> FitPolicy:
        try:
            return FitPolicy(value)
        except (TypeError, ValueError) as error:
            raise ValueError("fit policy must be first, best, or worst") from error


def create_application(**kwargs: object) -> HeapAllocator:
    """Registry factory for :class:`HeapAllocator`."""
    return HeapAllocator(**kwargs)  # type: ignore[arg-type]
