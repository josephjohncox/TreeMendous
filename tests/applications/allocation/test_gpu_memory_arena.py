"""GPU arena application contracts."""

from dataclasses import replace

import pytest

from tests.oracles.applications.allocation.gpu_memory_arena import eligible
from treemendous.applications._shared.allocation import ForeignAllocationError
from treemendous.applications.allocation.gpu_arena import GPUMemoryArena
from treemendous.domain import Span


def test_alignment_stream_ownership_and_deferred_reclamation() -> None:
    arena = GPUMemoryArena(4096, base_address=4096, device_alignment=256)
    first = arena.allocate(300, stream="s0", alignment=512)
    second = arena.allocate(200, stream="s1")
    assert first.device_range.start % 512 == 0
    arena.defer_free(first, stream="s0", completion_epoch=3)
    arena.defer_free(second, stream="s1", completion_epoch=2)
    expected = eligible((("s0", 3, first.handle.allocation_id),), "s0", 3)
    reclaimed = arena.advance_completion("s0", 3)
    assert tuple(item.handle.allocation_id for item in reclaimed) == expected
    assert arena.snapshot().diagnostics.allocated_space == second.handle.size


def test_wrong_stream_and_epoch_regression_are_failure_atomic() -> None:
    arena = GPUMemoryArena(1024)
    buffer = arena.allocate(128, stream="owner")
    before = arena.snapshot()
    with pytest.raises(ForeignAllocationError):
        arena.defer_free(buffer, stream="other", completion_epoch=1)
    assert arena.snapshot() == before
    arena.advance_completion("owner", 2)
    with pytest.raises(ValueError, match="backwards"):
        arena.advance_completion("owner", 1)


def test_restore_rejects_forged_reserved_geometry_atomically() -> None:
    arena = GPUMemoryArena(1024, base_address=4096)
    checkpoint = arena.checkpoint()
    forged_allocator = replace(
        checkpoint.allocator,
        reserved_ranges=(Span(4096, 4352),),
        free_ranges=(Span(4352, 5120),),
    )
    before = arena.snapshot()

    with pytest.raises(ValueError, match="configured allocator geometry"):
        arena.restore(replace(checkpoint, allocator=forged_allocator))

    assert arena.snapshot() == before
