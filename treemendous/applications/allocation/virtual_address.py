"""Page-granular virtual address-space mapping with guard pages."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from threading import RLock

from treemendous.applications._shared.allocation import (
    AllocationHandle,
    AllocatorCheckpoint,
    ContiguousAllocator,
    ForeignAllocationError,
    FragmentationDiagnostics,
)
from treemendous.domain import Span, validate_coordinate, validate_length


@dataclass(frozen=True)
class VirtualMapping:
    """One mapping and the guard-page reservation surrounding it."""

    handle: AllocationHandle
    owner: Hashable
    address: int
    mapped_length: int
    requested_length: int
    page_size: int
    guard_pages: int
    movable: bool

    @property
    def byte_range(self) -> Span:
        return Span(self.address, self.address + self.mapped_length)

    @property
    def reserved_pages(self) -> Span:
        return self.handle.span


@dataclass(frozen=True)
class AddressSpaceSnapshot:
    """Immutable mappings and page-level free-space diagnostics."""

    page_size: int
    mappings: tuple[VirtualMapping, ...]
    free_page_ranges: tuple[Span, ...]
    diagnostics: FragmentationDiagnostics


@dataclass(frozen=True)
class AddressSpaceCheckpoint:
    """Restorable address-space state."""

    allocator: AllocatorCheckpoint
    mappings: tuple[VirtualMapping, ...]


class VirtualAddressSpace:
    """Manage fixed or movable page mappings with symmetric guard pages."""

    def __init__(
        self,
        total_pages: int,
        *,
        page_size: int = 4096,
        base_address: int = 0,
        reserved_pages: tuple[Span, ...] = (),
    ) -> None:
        validate_length(total_pages)
        validate_length(page_size)
        validate_coordinate(base_address, "base_address")
        if base_address % page_size:
            raise ValueError("base_address must be page aligned")
        self._base_page = base_address // page_size
        self._page_size = page_size
        self._allocator = ContiguousAllocator(
            (self._base_page, self._base_page + total_pages), reserved=reserved_pages
        )
        self._reserved_pages = self._allocator.snapshot().reserved_ranges
        self._mappings: dict[int, VirtualMapping] = {}
        self._lock = RLock()

    def map(
        self,
        length: int,
        *,
        owner: Hashable,
        address: int | None = None,
        guard_pages: int = 1,
        movable: bool = True,
    ) -> VirtualMapping:
        """Create a page-rounded mapping, optionally at an exact address."""
        validate_length(length)
        validate_coordinate(guard_pages, "guard_pages")
        if guard_pages < 0:
            raise ValueError("guard_pages must be nonnegative")
        mapped_pages = (length + self._page_size - 1) // self._page_size
        reserved_count = mapped_pages + 2 * guard_pages
        with self._lock:
            if address is None:
                handle = self._allocator.allocate(reserved_count, owner=owner)
            else:
                validate_coordinate(address, "address")
                if address % self._page_size:
                    raise ValueError("address must be page aligned")
                payload_page = address // self._page_size
                handle = self._allocator.reserve(
                    payload_page - guard_pages, reserved_count, owner=owner
                )
            mapping = self._make_mapping(
                handle, owner, length, mapped_pages, guard_pages, movable
            )
            self._mappings[handle.allocation_id] = mapping
            return mapping

    def unmap(self, mapping: VirtualMapping, *, owner: Hashable) -> None:
        """Release a live mapping and both guard regions."""
        if not isinstance(mapping, VirtualMapping):
            raise TypeError("mapping must be a VirtualMapping")
        with self._lock:
            self._require_live(mapping)
            self._allocator.free(mapping.handle, owner=owner)
            del self._mappings[mapping.handle.allocation_id]

    def relocate(
        self,
        mapping: VirtualMapping,
        *,
        owner: Hashable,
        address: int | None = None,
    ) -> VirtualMapping:
        """Atomically move a movable mapping to an exact or selected address."""
        if not isinstance(mapping, VirtualMapping):
            raise TypeError("mapping must be a VirtualMapping")
        with self._lock:
            self._require_live(mapping)
            if owner != mapping.owner:
                raise ForeignAllocationError("mapping belongs to another owner")
            if not mapping.movable:
                raise ValueError("fixed mapping cannot be relocated")
            if address == mapping.address:
                return mapping
            checkpoint = self._allocator.checkpoint()
            old_mappings = dict(self._mappings)
            try:
                self._allocator.free(mapping.handle, owner=owner)
                del self._mappings[mapping.handle.allocation_id]
                replacement = self.map(
                    mapping.requested_length,
                    owner=owner,
                    address=address,
                    guard_pages=mapping.guard_pages,
                    movable=True,
                )
            except Exception:
                self._allocator.restore(checkpoint)
                self._mappings = old_mappings
                raise
            return replacement

    def snapshot(self) -> AddressSpaceSnapshot:
        """Return page geometry, mappings, and fragmentation diagnostics."""
        with self._lock:
            state = self._allocator.snapshot()
            return AddressSpaceSnapshot(
                page_size=self._page_size,
                mappings=tuple(
                    sorted(self._mappings.values(), key=lambda item: item.address)
                ),
                free_page_ranges=state.free_ranges,
                diagnostics=state.diagnostics,
            )

    def checkpoint(self) -> AddressSpaceCheckpoint:
        """Capture complete address-space state."""
        with self._lock:
            return AddressSpaceCheckpoint(
                self._allocator.checkpoint(), self.snapshot().mappings
            )

    def restore(self, checkpoint: AddressSpaceCheckpoint) -> None:
        """Atomically restore a structurally valid local checkpoint."""
        if not isinstance(checkpoint, AddressSpaceCheckpoint):
            raise TypeError("checkpoint must be an AddressSpaceCheckpoint")
        self._allocator.validate_checkpoint_geometry(
            checkpoint.allocator, reserved_ranges=self._reserved_pages
        )
        handles = {record.handle for record in checkpoint.allocator.records}
        staged: dict[int, VirtualMapping] = {}
        for mapping in checkpoint.mappings:
            if not isinstance(mapping, VirtualMapping):
                raise TypeError("checkpoint mappings must be VirtualMapping values")
            mapped_pages = mapping.mapped_length // self._page_size
            expected_address = (
                mapping.handle.start + mapping.guard_pages
            ) * self._page_size
            if (
                mapping.handle not in handles
                or mapping.owner != mapping.handle.owner
                or mapping.page_size != self._page_size
                or mapping.address != expected_address
                or mapping.address % self._page_size
                or mapping.requested_length <= 0
                or mapping.mapped_length
                != (
                    (mapping.requested_length + self._page_size - 1)
                    // self._page_size
                )
                * self._page_size
                or mapping.mapped_length % self._page_size
                or mapping.guard_pages < 0
                or mapping.handle.size != mapped_pages + 2 * mapping.guard_pages
                or mapping.handle.allocation_id in staged
            ):
                raise ValueError("checkpoint contains invalid mapping metadata")
            staged[mapping.handle.allocation_id] = mapping
        if len(staged) != len(handles):
            raise ValueError("checkpoint is missing mapping metadata")
        with self._lock:
            self._allocator.restore(checkpoint.allocator)
            self._mappings = staged

    def _make_mapping(
        self,
        handle: AllocationHandle,
        owner: Hashable,
        requested_length: int,
        mapped_pages: int,
        guard_pages: int,
        movable: bool,
    ) -> VirtualMapping:
        return VirtualMapping(
            handle=handle,
            owner=owner,
            address=(handle.start + guard_pages) * self._page_size,
            mapped_length=mapped_pages * self._page_size,
            requested_length=requested_length,
            page_size=self._page_size,
            guard_pages=guard_pages,
            movable=movable,
        )

    def _require_live(self, mapping: VirtualMapping) -> None:
        if self._mappings.get(mapping.handle.allocation_id) != mapping:
            from treemendous.applications._shared.allocation import StaleAllocationError

            raise StaleAllocationError("virtual mapping is stale or foreign")


def create_application(**kwargs: object) -> VirtualAddressSpace:
    """Registry factory for :class:`VirtualAddressSpace`."""
    return VirtualAddressSpace(**kwargs)  # type: ignore[arg-type]
