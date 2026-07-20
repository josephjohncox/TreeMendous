"""Adversarial public-contract coverage for allocation applications."""

from dataclasses import replace
from typing import Any, cast

import pytest

from treemendous.applications._shared.allocation import (
    ContiguousAllocator,
    FitPolicy,
    ForeignAllocationError,
    StaleAllocationError,
)
from treemendous.applications._shared.ring_sequences import RingSequenceTracker
from treemendous.applications.allocation import (
    cdn_cache,
    database_pages,
    disk_blocks,
    gpu_arena,
    heap,
    multipart_upload,
    ring_buffer,
    virtual_address,
)
from treemendous.applications.allocation.cdn_cache import (
    CacheCheckpoint,
    CDNByteRangeCache,
)
from treemendous.applications.allocation.database_pages import (
    DatabaseCheckpoint,
    DatabasePageAllocator,
)
from treemendous.applications.allocation.disk_blocks import (
    DiskBlockAllocator,
    DiskCheckpoint,
)
from treemendous.applications.allocation.gpu_arena import (
    GPUArenaCheckpoint,
    GPUMemoryArena,
)
from treemendous.applications.allocation.heap import HeapAllocator, HeapCheckpoint
from treemendous.applications.allocation.multipart_upload import (
    MultipartCheckpoint,
    MultipartUploadTracker,
    PartConflictError,
)
from treemendous.applications.allocation.ring_buffer import (
    RingBuffer,
    RingBufferCheckpoint,
    RingEmptyError,
)
from treemendous.applications.allocation.virtual_address import (
    AddressSpaceCheckpoint,
    VirtualAddressSpace,
)
from treemendous.domain import Span


def test_application_factories_preserve_public_geometry() -> None:
    """Every registry factory constructs the documented allocation engine."""
    assert (
        gpu_arena.create_application(capacity=256).snapshot().diagnostics.total_space
        == 256
    )
    assert (
        virtual_address.create_application(total_pages=4).snapshot().page_size == 4096
    )
    assert heap.create_application(capacity=32).snapshot().capacity == 32
    assert (
        database_pages.create_application(total_pages=4).snapshot().tablespace == "main"
    )
    assert disk_blocks.create_application(total_blocks=4).snapshot().total_blocks == 4
    assert cdn_cache.create_application(object_size=4).snapshot().object_size == 4
    assert ring_buffer.create_application(capacity=4).snapshot().capacity == 4
    assert (
        multipart_upload.create_application(object_size=4, part_size=2)
        .snapshot()
        .diagnostics.total_parts
        == 2
    )


def test_gpu_validation_deferred_idempotency_and_immediate_reclamation() -> None:
    with pytest.raises(ValueError, match="device alignment"):
        GPUMemoryArena(256, base_address=1)
    with pytest.raises(ValueError, match="power of two"):
        GPUMemoryArena(256, device_alignment=3)

    arena = GPUMemoryArena(1024, device_alignment=256)
    with pytest.raises(ValueError, match="less than"):
        arena.allocate(32, stream="s", alignment=128)
    with pytest.raises(ValueError, match="power of two"):
        arena.allocate(32, stream="s", alignment=0)
    with pytest.raises(TypeError, match="GPUBuffer"):
        arena.defer_free(cast(Any, object()), stream="s", completion_epoch=0)
    buffer = arena.allocate(64, stream="s")
    with pytest.raises(ValueError, match="nonnegative"):
        arena.defer_free(buffer, stream="s", completion_epoch=-1)
    deferred = arena.defer_free(buffer, stream="s", completion_epoch=3)
    assert deferred.buffer == buffer
    with pytest.raises(ValueError, match="already"):
        arena.defer_free(buffer, stream="s", completion_epoch=4)
    assert not arena.advance_completion("s", 2)
    assert arena.snapshot().deferred_frees[0] == deferred
    assert arena.advance_completion("s", 3)[0] == buffer
    with pytest.raises(StaleAllocationError):
        arena.defer_free(buffer, stream="s", completion_epoch=4)
    with pytest.raises(ValueError, match="nonnegative"):
        arena.advance_completion("s", -1)

    immediate = arena.allocate(64, stream="s")
    result = arena.defer_free(immediate, stream="s", completion_epoch=3)
    assert result.buffer == immediate
    assert not arena.snapshot().live_buffers


def test_gpu_advance_completion_rolls_back_dependency_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arena = GPUMemoryArena(512)
    buffer = arena.allocate(64, stream="stream")
    arena.defer_free(buffer, stream="stream", completion_epoch=1)
    before = arena.snapshot()

    def fail_free(self: ContiguousAllocator, handle: object, *, owner: object) -> None:
        raise RuntimeError("simulated device release failure")

    monkeypatch.setattr(ContiguousAllocator, "free", fail_free)
    with pytest.raises(RuntimeError, match="simulated"):
        arena.advance_completion("stream", 1)
    assert arena.snapshot() == before


def test_gpu_checkpoint_validation_and_valid_restore_are_atomic() -> None:
    arena = GPUMemoryArena(1024)
    live = arena.allocate(64, stream="live")
    deferred_buffer = arena.allocate(64, stream="deferred")
    arena.defer_free(deferred_buffer, stream="deferred", completion_epoch=5)
    arena.advance_completion("deferred", 2)
    checkpoint = arena.checkpoint()
    expected = arena.snapshot()
    arena.allocate(64, stream="temporary")
    arena.restore(checkpoint)
    assert arena.snapshot() == expected
    before = arena.snapshot()

    invalid_buffer = replace(live, stream="foreign")
    duplicate_deferred = checkpoint.deferred_frees * 2
    cases: tuple[tuple[object, type[Exception], str], ...] = (
        (object(), TypeError, "GPUArenaCheckpoint"),
        (replace(checkpoint, buffers=(cast(Any, object()),)), TypeError, "GPUBuffer"),
        (
            replace(
                checkpoint,
                buffers=(invalid_buffer, deferred_buffer),
            ),
            ValueError,
            "GPU buffer metadata",
        ),
        (replace(checkpoint, buffers=()), ValueError, "missing GPU buffer"),
        (
            replace(checkpoint, deferred_frees=(cast(Any, object()),)),
            TypeError,
            "deferred frees",
        ),
        (
            replace(checkpoint, deferred_frees=duplicate_deferred),
            ValueError,
            "deferred-free metadata",
        ),
        (
            replace(checkpoint, completed_epochs=(("deferred", -1),)),
            ValueError,
            "stream epochs",
        ),
        (
            replace(
                checkpoint,
                completed_epochs=(("deferred", 1), ("deferred", 2)),
            ),
            ValueError,
            "stream epochs",
        ),
        (
            replace(checkpoint, completed_epochs=(("deferred", 5),)),
            ValueError,
            "already-completed",
        ),
    )
    for forged, error, match in cases:
        with pytest.raises(error, match=match):
            arena.restore(cast(GPUArenaCheckpoint, forged))
        assert arena.snapshot() == before


def test_virtual_address_validation_unmapping_and_successful_relocation() -> None:
    with pytest.raises(ValueError, match="page aligned"):
        VirtualAddressSpace(8, page_size=1024, base_address=1)
    space = VirtualAddressSpace(12, page_size=1024)
    with pytest.raises(ValueError, match="nonnegative"):
        space.map(512, owner="p", guard_pages=-1)
    with pytest.raises(ValueError, match="page aligned"):
        space.map(512, owner="p", address=1)
    selected = space.map(512, owner="p", guard_pages=0)
    assert selected.address == 0
    assert space.relocate(selected, owner="p", address=selected.address) is selected
    with pytest.raises(TypeError, match="VirtualMapping"):
        space.unmap(cast(Any, object()), owner="p")
    with pytest.raises(TypeError, match="VirtualMapping"):
        space.relocate(cast(Any, object()), owner="p")
    relocated = space.relocate(selected, owner="p")
    assert relocated != selected
    assert relocated.requested_length == selected.requested_length
    with pytest.raises(StaleAllocationError):
        space.unmap(selected, owner="p")
    with pytest.raises(ForeignAllocationError):
        space.unmap(relocated, owner="other")
    assert space.snapshot().mappings[0] == relocated


def test_virtual_address_checkpoint_metadata_validation_is_atomic() -> None:
    space = VirtualAddressSpace(12, page_size=1024)
    mapping = space.map(1500, owner="owner", guard_pages=1)
    checkpoint = space.checkpoint()
    expected = space.snapshot()
    space.map(1024, owner="temporary", guard_pages=0)
    space.restore(checkpoint)
    assert space.snapshot() == expected
    before = space.snapshot()

    invalid_mapping = replace(mapping, requested_length=0)
    cases: tuple[tuple[object, type[Exception], str], ...] = (
        (object(), TypeError, "AddressSpaceCheckpoint"),
        (
            replace(checkpoint, mappings=(cast(Any, object()),)),
            TypeError,
            "VirtualMapping",
        ),
        (
            replace(checkpoint, mappings=(invalid_mapping,)),
            ValueError,
            "mapping metadata",
        ),
        (
            replace(checkpoint, mappings=(mapping, mapping)),
            ValueError,
            "mapping metadata",
        ),
        (replace(checkpoint, mappings=()), ValueError, "missing mapping"),
    )
    for forged, error, match in cases:
        with pytest.raises(error, match=match):
            space.restore(cast(AddressSpaceCheckpoint, forged))
        assert space.snapshot() == before


def test_heap_fit_policies_validation_ownership_and_staleness() -> None:
    for kwargs, match in (
        ({"header_size": -1}, "header_size"),
        ({"redzone_size": -1}, "redzone_size"),
        ({"default_alignment": 3}, "power of two"),
        ({"fit_policy": "invalid"}, "fit policy"),
    ):
        with pytest.raises(ValueError, match=match):
            HeapAllocator(64, **kwargs)

    best = HeapAllocator(64, header_size=0, fit_policy=FitPolicy.BEST)
    best_block = best.allocate(8, owner="best", policy="best")
    assert best_block.owner == "best"
    worst = HeapAllocator(64, header_size=0)
    assert worst.allocate(8, owner="worst", policy="worst").owner == "worst"
    with pytest.raises(ValueError, match="fit policy"):
        best.allocate(8, owner="x", policy="invalid")
    with pytest.raises(ValueError, match="power of two"):
        best.allocate(8, owner="x", alignment=6)
    with pytest.raises(TypeError, match="HeapBlock"):
        best.free(cast(Any, object()), owner="best")
    with pytest.raises(ForeignAllocationError):
        best.free(best_block, owner="foreign")
    best.free(best_block, owner="best")
    with pytest.raises(StaleAllocationError):
        best.free(best_block, owner="best")


def test_heap_checkpoint_layout_validation_is_atomic() -> None:
    allocator = HeapAllocator(64, header_size=4, redzone_size=2)
    block = allocator.allocate(8, owner="owner")
    checkpoint = allocator.checkpoint()
    allocator.free(block, owner="owner")
    allocator.restore(checkpoint)
    assert allocator.checkpoint() == checkpoint
    before = allocator.snapshot()

    cases: tuple[tuple[object, type[Exception], str], ...] = (
        (object(), TypeError, "HeapCheckpoint"),
        (
            replace(checkpoint, blocks=(cast(Any, object()),)),
            ValueError,
            "block metadata",
        ),
        (
            replace(checkpoint, blocks=(replace(block, requested_size=7),)),
            ValueError,
            "invalid heap layout",
        ),
        (
            replace(checkpoint, blocks=(block, block)),
            ValueError,
            "invalid heap layout",
        ),
        (replace(checkpoint, blocks=()), ValueError, "missing heap block"),
    )
    for forged, error, match in cases:
        with pytest.raises(error, match=match):
            allocator.restore(cast(HeapCheckpoint, forged))
        assert allocator.snapshot() == before


def test_database_validation_stale_extents_and_history_normalization() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        DatabasePageAllocator(4, tablespace="")
    for metadata_pages in (-1, 4):
        with pytest.raises(ValueError, match="within"):
            DatabasePageAllocator(4, metadata_pages=metadata_pages)

    pages = DatabasePageAllocator(8, metadata_pages=0)
    first = pages.allocate_pages("first", 2)
    second = pages.allocate_pages("second", 2)
    with pytest.raises(TypeError, match="PageExtent"):
        pages.free_pages(cast(Any, object()), table_id="first")
    with pytest.raises(ForeignAllocationError):
        pages.free_pages(first, table_id="foreign")
    pages.free_pages(first, table_id="first")
    with pytest.raises(StaleAllocationError):
        pages.free_pages(first, table_id="first")
    pages.free_pages(second, table_id="second")
    assert pages.snapshot().diagnostics.pages_ever_freed == 4


def test_database_checkpoint_metadata_and_counters_are_atomic() -> None:
    pages = DatabasePageAllocator(10, metadata_pages=1)
    extent = pages.allocate_pages("table", 2)
    checkpoint = pages.checkpoint()
    before = pages.snapshot()
    cases: tuple[tuple[object, type[Exception], str], ...] = (
        (object(), TypeError, "DatabaseCheckpoint"),
        (
            replace(checkpoint, reuse_allocations=-1),
            ValueError,
            "reuse_allocations",
        ),
        (
            replace(checkpoint, freed_history=(cast(Any, "bad"),)),
            TypeError,
            "Span",
        ),
        (
            replace(checkpoint, freed_history=(Span(3, 5), Span(5, 6))),
            ValueError,
            "normalized",
        ),
        (
            replace(checkpoint, extents=(cast(Any, object()),)),
            ValueError,
            "extent metadata",
        ),
        (
            replace(checkpoint, extents=(replace(extent, table_id="other"),)),
            ValueError,
            "extent metadata",
        ),
        (replace(checkpoint, extents=()), ValueError, "missing page extent"),
    )
    for forged, error, match in cases:
        with pytest.raises(error, match=match):
            pages.restore(cast(DatabaseCheckpoint, forged))
        assert pages.snapshot() == before


def test_disk_validation_policy_staleness_and_extent_filtering() -> None:
    for metadata_blocks in (-1, 4):
        with pytest.raises(ValueError, match="within"):
            DiskBlockAllocator(4, metadata_blocks=metadata_blocks)
    with pytest.raises(ValueError, match="fit policy"):
        DiskBlockAllocator(4, fit_policy="invalid")

    disk = DiskBlockAllocator(12, metadata_blocks=0, fit_policy="first")
    first = disk.allocate_extent("first", 2, policy="worst")
    disk.allocate_extent("second", 2)
    assert disk.snapshot().metadata_blocks is None
    assert disk.extents_for("first")[0] == first
    with pytest.raises(TypeError, match="FileExtent"):
        disk.free_extent(cast(Any, object()), file_id="first")
    with pytest.raises(ForeignAllocationError):
        disk.free_extent(first, file_id="foreign")
    disk.free_extent(first, file_id="first")
    with pytest.raises(StaleAllocationError):
        disk.free_extent(first, file_id="first")


def test_disk_checkpoint_extent_validation_is_atomic() -> None:
    disk = DiskBlockAllocator(8, metadata_blocks=1)
    extent = disk.allocate_extent("file", 2)
    checkpoint = disk.checkpoint()
    before = disk.snapshot()
    cases: tuple[tuple[object, type[Exception], str], ...] = (
        (object(), TypeError, "DiskCheckpoint"),
        (
            replace(checkpoint, extents=(cast(Any, object()),)),
            ValueError,
            "extent metadata",
        ),
        (
            replace(checkpoint, extents=(replace(extent, block_size=1),)),
            ValueError,
            "extent metadata",
        ),
        (
            replace(checkpoint, extents=(extent, extent)),
            ValueError,
            "extent metadata",
        ),
        (replace(checkpoint, extents=()), ValueError, "missing file extent"),
    )
    for forged, error, match in cases:
        with pytest.raises(error, match=match):
            disk.restore(cast(DiskCheckpoint, forged))
        assert disk.snapshot() == before


def test_cdn_validation_eviction_staleness_and_request_bounds() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        CDNByteRangeCache(10, object_id="")
    cache = CDNByteRangeCache(10)
    segment = cache.cache_segment(2, 4, cache_key="segment")
    with pytest.raises(TypeError, match="CachedSegment"):
        cache.evict(cast(Any, object()))
    with pytest.raises(ValueError, match="contained"):
        cache.request_coverage(-1, 2)
    with pytest.raises(ValueError, match="contained"):
        cache.request_coverage(9, 2)
    cache.evict(segment)
    with pytest.raises(StaleAllocationError):
        cache.evict(segment)
    assert cache.snapshot().diagnostics.evictions == 1


def test_cdn_checkpoint_metadata_and_eviction_counter_are_atomic() -> None:
    cache = CDNByteRangeCache(10)
    segment = cache.cache_segment(0, 5, cache_key="key")
    checkpoint = cache.checkpoint()
    before = cache.snapshot()
    cases: tuple[tuple[object, type[Exception], str], ...] = (
        (object(), TypeError, "CacheCheckpoint"),
        (replace(checkpoint, evictions=-1), ValueError, "nonnegative"),
        (
            replace(checkpoint, segments=(replace(segment, cache_key="other"),)),
            ValueError,
            "segment metadata",
        ),
        (
            replace(checkpoint, segments=(segment, segment)),
            ValueError,
            "segment metadata",
        ),
        (replace(checkpoint, segments=()), ValueError, "missing cached-segment"),
    )
    for forged, error, match in cases:
        with pytest.raises(error, match=match):
            cache.restore(cast(CacheCheckpoint, forged))
        assert cache.snapshot() == before


def test_ring_validation_epoch_hints_empty_consumption_and_empty_restore() -> None:
    for kwargs, match in (
        ({"sequence_modulus": 1}, "at least"),
        ({"sequence_modulus": 3}, "at least"),
        ({"sequence_modulus": 4, "initial_sequence": 4}, "modular domain"),
        ({"full_policy": "invalid"}, "full_policy"),
    ):
        with pytest.raises(ValueError, match=match):
            RingBuffer(4, **kwargs)

    ring = RingBuffer(4, sequence_modulus=8)
    with pytest.raises(ValueError, match="nonnegative"):
        ring.produce(1, epoch_hint=-1)
    with pytest.raises(ValueError, match="producer cursor"):
        ring.produce(1, epoch_hint=1)
    with pytest.raises(RingEmptyError):
        ring.consume(1)
    empty = ring.checkpoint()
    ring.produce(1)
    ring.restore(empty)
    assert ring.snapshot().occupancy == 0
    with pytest.raises(TypeError, match="RingBufferCheckpoint"):
        ring.restore(cast(RingBufferCheckpoint, object()))
    forged_sequences = replace(empty.sequences, received_ranges=(Span(0, 1),))
    with pytest.raises(ValueError, match="sequence history"):
        ring.restore(replace(empty, sequences=forged_sequences))


def test_ring_production_rolls_back_tracker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ring = RingBuffer(4, sequence_modulus=8)
    before = ring.snapshot()

    def fail_observe(
        self: RingSequenceTracker, sequence: int, *, epoch: int | None = None
    ) -> object:
        raise RuntimeError("simulated tracker failure")

    monkeypatch.setattr(RingSequenceTracker, "observe", fail_observe)
    with pytest.raises(RuntimeError, match="simulated"):
        ring.produce(2)
    assert ring.snapshot() == before


def test_multipart_validation_retry_intent_and_part_bounds() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        MultipartUploadTracker(10, 5, upload_id="")
    upload = MultipartUploadTracker(11, 5)
    with pytest.raises(ValueError, match="outside"):
        upload.part_range(0)
    with pytest.raises(ValueError, match="outside"):
        upload.part_range(4)
    with pytest.raises(ValueError, match="nonempty"):
        upload.complete_part(1, "")
    with pytest.raises(ValueError, match="expected"):
        upload.complete_part(1, "etag", size=4)
    with pytest.raises(PartConflictError, match="has not completed"):
        upload.complete_part(1, "etag", retry=True)
    assert upload.snapshot().diagnostics.completed_parts == 0


def test_multipart_checkpoint_metadata_validation_is_atomic() -> None:
    upload = MultipartUploadTracker(11, 5)
    part = upload.complete_part(1, "etag")
    checkpoint = upload.checkpoint()
    before = upload.snapshot()
    cases: tuple[tuple[object, type[Exception], str], ...] = (
        (object(), TypeError, "MultipartCheckpoint"),
        (
            replace(checkpoint, completed=(cast(Any, object()),)),
            TypeError,
            "CompletedPart",
        ),
        (
            replace(checkpoint, completed=(replace(part, attempt=0),)),
            ValueError,
            "part metadata",
        ),
        (
            replace(checkpoint, completed=(part, part)),
            ValueError,
            "part metadata",
        ),
        (replace(checkpoint, completed=()), ValueError, "missing completed-part"),
    )
    for forged, error, match in cases:
        with pytest.raises(error, match=match):
            upload.restore(cast(MultipartCheckpoint, forged))
        assert upload.snapshot() == before
