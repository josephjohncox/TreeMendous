"""Heap application contracts."""

from dataclasses import replace

import pytest

from tests.oracles.applications.allocation.heap_free_space import place
from treemendous.applications._shared.allocation import AllocationUnavailableError
from treemendous.applications.allocation.heap import HeapAllocator
from treemendous.domain import Span


def test_payload_alignment_overhead_and_oracle() -> None:
    heap = HeapAllocator(64, base_address=3, header_size=5, redzone_size=2)
    expected = place(heap.snapshot().free_ranges, 9, 5, 2, 16)
    block = heap.allocate(9, owner="request", alignment=16)
    assert expected is not None
    expected_raw, expected_payload = expected
    assert block.raw_handle.span == expected_raw
    assert block.payload == expected_payload
    assert block.payload.start % 16 == 0
    assert block.raw_handle.size == 18


def test_checkpoint_and_failure_atomicity() -> None:
    heap = HeapAllocator(32, header_size=4, redzone_size=2, default_alignment=8)
    retained = heap.allocate(8, owner="a")
    checkpoint = heap.checkpoint()
    heap.free(retained, owner="a")
    heap.restore(checkpoint)
    before = heap.snapshot()
    with pytest.raises(AllocationUnavailableError):
        heap.allocate(100, owner="b")
    assert heap.snapshot() == before
    assert before.diagnostics.allocated_space == retained.raw_handle.size


def test_restore_rejects_forged_reserved_geometry_atomically() -> None:
    heap = HeapAllocator(32)
    checkpoint = heap.checkpoint()
    forged_allocator = replace(
        checkpoint.allocator,
        reserved_ranges=(Span(0, 4),),
        free_ranges=(Span(4, 32),),
    )
    before = heap.snapshot()

    with pytest.raises(ValueError, match="configured allocator geometry"):
        heap.restore(replace(checkpoint, allocator=forged_allocator))

    assert heap.snapshot() == before
