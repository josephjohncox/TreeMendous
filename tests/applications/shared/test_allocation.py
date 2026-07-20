"""Contracts for the private contiguous allocation kernel."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import pytest

from treemendous.applications._shared.allocation import (
    AllocationConflictError,
    AllocationUnavailableError,
    ContiguousAllocator,
    FitPolicy,
    ForeignAllocationError,
    StaleAllocationError,
)
from treemendous.domain import ManagedDomain, Span


def test_fit_policies_inspect_chunks_with_deterministic_ties() -> None:
    holes = (Span(12, 20), Span(30, 40))

    first = ContiguousAllocator((0, 60), reserved=holes)
    assert first.allocate(8, owner="first", policy=FitPolicy.FIRST).span == Span(0, 8)

    best = ContiguousAllocator((0, 60), reserved=holes)
    assert best.allocate(8, owner="best", policy=FitPolicy.BEST).span == Span(20, 28)

    worst = ContiguousAllocator((0, 60), reserved=holes)
    assert worst.allocate(8, owner="worst", policy=FitPolicy.WORST).span == Span(
        40, 48
    )

    tied = ContiguousAllocator((0, 30), reserved=(Span(10, 20),))
    assert tied.allocate(2, owner="tie", policy="best").start == 0
    tied = ContiguousAllocator((0, 30), reserved=(Span(10, 20),))
    assert tied.allocate(2, owner="tie", policy="worst").start == 0


def test_alignment_is_correct_for_negative_coordinates_and_preserves_padding() -> None:
    allocator = ContiguousAllocator((-10, 10))
    handle = allocator.allocate(3, owner="negative", alignment=4)

    assert handle.span == Span(-8, -5)
    expected_free = (Span(-10, -8), Span(-5, 10))
    assert allocator.snapshot().free_ranges == expected_free


def test_exact_reserve_is_atomic_and_removes_only_requested_extent() -> None:
    allocator = ContiguousAllocator((0, 20))
    handle = allocator.reserve(8, 3, owner="device", alignment=4)

    assert handle.span == Span(8, 11)
    expected_free = (Span(0, 8), Span(11, 20))
    assert allocator.snapshot().free_ranges == expected_free
    before = allocator.snapshot()
    with pytest.raises(AllocationUnavailableError, match="entirely free"):
        allocator.reserve(7, 4, owner="other")
    assert allocator.snapshot() == before


def test_idempotency_is_owner_scoped_and_request_sensitive() -> None:
    allocator = ContiguousAllocator((0, 20))
    original = allocator.allocate(4, owner="a", idempotency_key="request")

    assert allocator.allocate(4, owner="a", idempotency_key="request") == original
    other_owner = allocator.allocate(4, owner="b", idempotency_key="request")
    assert other_owner != original
    with pytest.raises(AllocationConflictError, match="different request"):
        allocator.allocate(5, owner="a", idempotency_key="request")

    allocator.free(original, owner="a")
    replacement = allocator.allocate(4, owner="a", idempotency_key="request")
    assert replacement.allocation_id != original.allocation_id


def test_free_requires_owner_and_rejects_cross_owner_snapshot_handles() -> None:
    allocator = ContiguousAllocator((0, 10))
    foreign_allocator = ContiguousAllocator((0, 10))
    handle = allocator.allocate(2, owner="owner")
    exposed_handle = allocator.snapshot().allocations[0]

    with pytest.raises(TypeError, match="owner"):
        allocator.free(exposed_handle)
    with pytest.raises(ForeignAllocationError, match="another allocator"):
        foreign_allocator.free(handle, owner="owner")
    with pytest.raises(ForeignAllocationError, match="another owner"):
        allocator.free(exposed_handle, owner="intruder")
    allocator.free(handle, owner="owner")
    with pytest.raises(StaleAllocationError, match="stale"):
        allocator.free(handle, owner="owner")


def test_reserved_holes_snapshots_and_fragmentation_diagnostics() -> None:
    allocator = ContiguousAllocator((0, 30), reserved=(Span(5, 10),))
    allocator.reserve_hole((20, 25))
    handle = allocator.allocate(5, owner="owner")
    snapshot = allocator.snapshot()

    assert handle.span == Span(0, 5)
    expected_reserved = (Span(5, 10), Span(20, 25))
    expected_free = (Span(10, 20), Span(25, 30))
    expected_allocations = (handle,)
    assert snapshot.reserved_ranges == expected_reserved
    assert snapshot.free_ranges == expected_free
    assert snapshot.allocations == expected_allocations
    assert snapshot.diagnostics.total_space == 30
    assert snapshot.diagnostics.allocated_space == 5
    assert snapshot.diagnostics.reserved_space == 10
    assert snapshot.diagnostics.total_free == 15
    assert snapshot.diagnostics.free_chunks == 2
    assert snapshot.diagnostics.largest_free_chunk == 10
    assert snapshot.diagnostics.fragmentation == pytest.approx(1 / 3)


def test_checkpoint_restore_revives_state_without_reissuing_handle_ids() -> None:
    allocator = ContiguousAllocator((0, 20), reserved=(Span(10, 12),))
    retained = allocator.allocate(
        3, owner="owner", alignment=2, idempotency_key="retained"
    )
    checkpoint = allocator.checkpoint()
    later = allocator.allocate(2, owner="later")

    allocator.restore(checkpoint)

    expected_allocations = (retained,)
    assert allocator.snapshot().allocations == expected_allocations
    assert (
        allocator.allocate(
            3, owner="owner", alignment=2, idempotency_key="retained"
        )
        == retained
    )
    replacement = allocator.allocate(2, owner="later")
    assert replacement.allocation_id > later.allocation_id
    assert replacement != later
    with pytest.raises(StaleAllocationError):
        allocator.free(later, owner="later")
    expected_restored_allocations = (retained, replacement)
    assert allocator.snapshot().allocations == expected_restored_allocations
    allocator.free(retained, owner="owner")


def test_checkpoint_validation_is_failure_atomic() -> None:
    allocator = ContiguousAllocator((0, 12))
    allocator.allocate(3, owner="owner")
    checkpoint = allocator.checkpoint()
    malformed = replace(checkpoint, free_ranges=(Span(0, 12),))
    before = allocator.snapshot()

    with pytest.raises(ValueError, match="free geometry"):
        allocator.restore(malformed)
    assert allocator.snapshot() == before


def test_restore_rejects_foreign_and_structurally_invalid_checkpoints() -> None:
    allocator = ContiguousAllocator((0, 12))
    allocator.allocate(
        3,
        owner="owner",
        alignment=2,
        policy=FitPolicy.BEST,
        idempotency_key="key",
    )
    allocator.reserve(4, 2, owner="exact", alignment=2)
    checkpoint = allocator.checkpoint()
    policy_record, exact_record = checkpoint.records
    before = allocator.snapshot()

    invalid_checkpoints = [
        replace(checkpoint, domain=ManagedDomain((0, 13))),
        replace(checkpoint, next_allocation_id=0),
        replace(
            checkpoint,
            reserved_ranges=(Span(8, 10), Span(9, 11)),
        ),
        replace(checkpoint, reserved_ranges=(Span(20, 21),)),
        replace(
            checkpoint,
            records=(
                replace(
                    policy_record,
                    handle=replace(policy_record.handle, allocation_id=0),
                ),
                exact_record,
            ),
        ),
        replace(checkpoint, records=(policy_record, policy_record)),
        replace(
            checkpoint,
            records=(replace(policy_record, size=4), exact_record),
        ),
        replace(
            checkpoint,
            records=(
                replace(
                    policy_record,
                    handle=replace(policy_record.handle, span=Span(1, 4)),
                ),
                exact_record,
            ),
        ),
        replace(
            checkpoint,
            records=(replace(policy_record, exact_start=0), exact_record),
        ),
        replace(
            checkpoint,
            records=(policy_record, replace(exact_record, exact_start=5)),
        ),
        replace(
            checkpoint,
            records=(
                replace(
                    policy_record,
                    handle=replace(policy_record.handle, span=Span(12, 15)),
                ),
                exact_record,
            ),
        ),
        replace(checkpoint, next_allocation_id=2),
        replace(
            checkpoint,
            idempotency=(*checkpoint.idempotency, *checkpoint.idempotency),
        ),
        replace(checkpoint, idempotency=(("owner", "key", 999),)),
        replace(checkpoint, idempotency=()),
    ]
    for malformed in invalid_checkpoints:
        with pytest.raises((TypeError, ValueError, ForeignAllocationError)):
            allocator.restore(malformed)
        assert allocator.snapshot() == before

    foreign = ContiguousAllocator((0, 12)).checkpoint()
    with pytest.raises(ForeignAllocationError, match="checkpoint"):
        allocator.restore(foreign)
    invalid_checkpoint: Any = object()
    with pytest.raises(TypeError, match="AllocatorCheckpoint"):
        allocator.restore(invalid_checkpoint)


def test_validation_and_failed_operations_leave_state_unchanged() -> None:
    allocator = ContiguousAllocator((0, 8))
    before = allocator.snapshot()

    with pytest.raises(ValueError, match="greater than zero"):
        allocator.allocate(0, owner="x")
    with pytest.raises(ValueError, match="alignment"):
        allocator.allocate(1, owner="x", alignment=0)
    with pytest.raises(ValueError, match="policy"):
        allocator.allocate(1, owner="x", policy="unknown")
    with pytest.raises(ValueError, match="alignment"):
        allocator.reserve(3, 1, owner="x", alignment=2)
    invalid_owner: Any = []
    with pytest.raises(TypeError, match="hashable"):
        allocator.allocate(1, owner=invalid_owner)
    assert allocator.snapshot() == before


def test_concurrent_allocations_are_serialized_without_overlap() -> None:
    allocator = ContiguousAllocator((0, 100))
    with ThreadPoolExecutor(max_workers=12) as executor:
        handles = list(
            executor.map(lambda owner: allocator.allocate(1, owner=owner), range(100))
        )

    assert sorted(handle.start for handle in handles) == list(range(100))
    assert allocator.diagnostics().total_free == 0
    with pytest.raises(AllocationUnavailableError):
        allocator.allocate(1, owner="overflow")
