"""Disk block application contracts."""

from dataclasses import replace

import pytest

from tests.oracles.applications.allocation.disk_block_allocation import first_extent
from treemendous.applications._shared.allocation import ForeignAllocationError
from treemendous.applications.allocation.disk_blocks import DiskBlockAllocator
from treemendous.domain import Span


def test_metadata_block_size_and_contiguous_extents() -> None:
    disk = DiskBlockAllocator(20, block_size=4096, metadata_blocks=2)
    expected = first_extent(20, {0, 1}, 5)
    extent = disk.allocate_extent("file-a", 5)
    assert extent.blocks == expected == Span(2, 7)
    assert extent.byte_range == Span(8192, 28672)
    extents = disk.extents_for("file-a")
    assert len(extents) == 1
    assert extents[0] == extent


def test_release_reuses_extent_and_checkpoint_restores() -> None:
    disk = DiskBlockAllocator(12, metadata_blocks=1)
    first = disk.allocate_extent("a", 3)
    checkpoint = disk.checkpoint()
    disk.free_extent(first, file_id="a")
    reused = disk.allocate_extent("b", 3)
    assert reused.blocks == first.blocks
    disk.restore(checkpoint)
    before = disk.snapshot()
    with pytest.raises(ForeignAllocationError):
        disk.free_extent(first, file_id="wrong")
    assert disk.snapshot() == before


def test_restore_rejects_forged_metadata_reservation_atomically() -> None:
    disk = DiskBlockAllocator(10, metadata_blocks=2)
    checkpoint = disk.checkpoint()
    forged_allocator = replace(
        checkpoint.allocator,
        reserved_ranges=(),
        free_ranges=(Span(0, 10),),
    )
    before = disk.snapshot()

    with pytest.raises(ValueError, match="configured allocator geometry"):
        disk.restore(replace(checkpoint, allocator=forged_allocator))

    assert disk.snapshot() == before
    assert disk.allocate_extent("file", 1).blocks == Span(2, 3)
