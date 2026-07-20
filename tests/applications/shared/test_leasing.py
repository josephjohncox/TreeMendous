"""Contracts for the private fenced numeric lease kernel."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications._shared.leasing import (
    ExpiredLeaseError,
    FenceValidator,
    ForeignLeaseError,
    LeasePool,
    LeaseRequestConflictError,
    LeaseState,
    LeaseUnavailableError,
    StaleLeaseError,
)
from treemendous.domain import Span


def test_deterministic_allocation_across_disjoint_spans_and_alignment() -> None:
    clock = LogicalClock(10)
    pool = LeasePool((Span(-5, 0), Span(10, 20)), clock=clock)

    first = pool.acquire("a", ttl=5, size=2, alignment=4)
    second = pool.acquire("b", ttl=5, size=2, alignment=4)
    exact = pool.acquire("c", ttl=5, exact_span=(16, 18), alignment=2)

    assert first.resource == Span(-4, -2)
    assert second.resource == Span(12, 14)
    assert exact.resource == Span(16, 18)
    assert first.token == 1
    assert second.token == 2
    assert exact.token == 3
    assert first.state is LeaseState.ACTIVE
    assert second.state is LeaseState.ACTIVE
    assert exact.state is LeaseState.ACTIVE


def test_acquire_validates_inputs_and_exact_span_atomically() -> None:
    clock = LogicalClock()
    pool = LeasePool((0, 10), clock=clock)
    before = pool.snapshot()

    with pytest.raises(ValueError, match="ttl"):
        pool.acquire("owner", ttl=0)
    with pytest.raises(ValueError, match="size"):
        pool.acquire("owner", ttl=1, size=0)
    with pytest.raises(ValueError, match="alignment"):
        pool.acquire("owner", ttl=1, alignment=0)
    with pytest.raises(ValueError, match="alignment"):
        pool.acquire("owner", ttl=1, exact_span=(1, 3), alignment=2)
    with pytest.raises(ValueError, match="size"):
        pool.acquire("owner", ttl=1, exact_span=(1, 3), size=1)

    with pytest.raises(LeaseUnavailableError, match="outside"):
        pool.acquire("owner", ttl=1, exact_span=(20, 21))
    after = pool.snapshot()
    assert not after.leases
    assert not before.leases
    assert len(after.available_spans) == 1
    assert after.available_spans[0] == Span(0, 10)
    assert after.available_spans == before.available_spans
    assert after.diagnostics.next_fencing_token == 1


def test_request_id_is_idempotent_and_conflicts_are_rejected() -> None:
    clock = LogicalClock()
    pool = LeasePool((0, 4), clock=clock)

    original = pool.acquire("owner", ttl=3, size=2, request_id="request-1")
    assert pool.acquire("owner", ttl=3, size=2, request_id="request-1") == original
    with pytest.raises(LeaseRequestConflictError):
        pool.acquire("other", ttl=3, size=2, request_id="request-1")
    with pytest.raises(LeaseRequestConflictError):
        pool.acquire("owner", ttl=4, size=2, request_id="request-1")

    assert pool.snapshot().diagnostics.issued_tokens == 1
    clock.advance(3)
    terminal = pool.acquire("owner", ttl=3, size=2, request_id="request-1")
    assert terminal.state is LeaseState.EXPIRED
    assert terminal.token == original.token


def test_renew_release_and_expire_reject_old_foreign_and_terminal_handles() -> None:
    clock = LogicalClock()
    pool = LeasePool((0, 4), clock=clock)
    other_pool = LeasePool((0, 4), clock=clock)
    original = pool.acquire("owner", ttl=4, size=2)

    clock.advance()
    renewed = pool.renew(original, ttl=5, owner="owner")
    assert renewed.revision == 2
    assert renewed.expires_at == 6
    with pytest.raises(StaleLeaseError, match="old"):
        pool.release(original)
    with pytest.raises(ForeignLeaseError, match="owner"):
        pool.release(renewed, owner="intruder")
    with pytest.raises(ForeignLeaseError, match="another pool"):
        other_pool.release(renewed)

    released = pool.release(renewed)
    assert released.state is LeaseState.RELEASED
    with pytest.raises(StaleLeaseError, match="released"):
        pool.release(released)

    expiring = pool.acquire("next", ttl=2, size=2)
    clock.advance(2)
    expired = pool.expire()
    assert len(expired) == 1
    assert expired[0] == replace(expiring, state=LeaseState.EXPIRED)
    assert not pool.expire()
    with pytest.raises(ExpiredLeaseError):
        pool.renew(expiring, ttl=1)


def test_reused_resources_always_receive_new_global_tokens() -> None:
    clock = LogicalClock()
    pool = LeasePool((0, 1), clock=clock)

    first = pool.acquire("a", ttl=1)
    clock.advance()
    pool.expire()
    second = pool.acquire("b", ttl=1)
    pool.release(second)
    third = pool.acquire("c", ttl=1)

    assert first.resource == second.resource == third.resource == Span(0, 1)
    assert first.token == 1
    assert second.token == 2
    assert third.token == 3


def test_snapshot_diagnostics_and_checkpoint_restore_preserve_state() -> None:
    clock = LogicalClock(5)
    pool = LeasePool((Span(0, 3), Span(10, 14)), clock=clock)
    first = pool.acquire("a", ttl=10, size=2, request_id="a-1")
    pool.acquire("b", ttl=2, exact_span=(10, 12))
    pool.release(first)
    clock.advance(2)

    snapshot = pool.snapshot()
    assert len(snapshot.available_spans) == 2
    assert snapshot.available_spans[0] == Span(0, 3)
    assert snapshot.available_spans[1] == Span(10, 14)
    assert snapshot.leases[0].state is LeaseState.RELEASED
    assert snapshot.leases[1].state is LeaseState.EXPIRED
    assert snapshot.diagnostics.total_capacity == 7
    assert snapshot.diagnostics.available_capacity == 7
    assert snapshot.diagnostics.active_leases == 0
    assert snapshot.diagnostics.expired_leases == 1
    assert snapshot.diagnostics.released_leases == 1

    restored = LeasePool.from_checkpoint(pool.checkpoint(), clock=clock)
    retry = restored.acquire("a", ttl=10, size=2, request_id="a-1")
    assert retry.state is LeaseState.RELEASED
    assert retry.token == first.token
    replacement = restored.acquire("c", ttl=3, size=3)
    assert replacement.resource == Span(0, 3)
    assert replacement.token == 3
    assert restored.pool_id == pool.pool_id


def test_restore_expires_elapsed_active_lease_without_reusing_token() -> None:
    clock = LogicalClock()
    pool = LeasePool((0, 2), clock=clock)
    original = pool.acquire("owner", ttl=2, size=2)
    checkpoint = pool.checkpoint()
    clock.advance(3)

    restored = LeasePool.from_checkpoint(checkpoint, clock=clock)
    restored_leases = restored.snapshot().leases
    assert len(restored_leases) == 1
    assert restored_leases[0] == replace(original, state=LeaseState.EXPIRED)
    assert restored.acquire("next", ttl=1, size=2).token == 2


def test_checkpoint_validation_rejects_token_regression_and_clock_regression() -> None:
    clock = LogicalClock(2)
    pool = LeasePool((0, 2), clock=clock)
    pool.acquire("owner", ttl=2)
    checkpoint = pool.checkpoint()

    with pytest.raises(ValueError, match="exceed"):
        LeasePool.from_checkpoint(
            replace(checkpoint, next_fencing_token=1), clock=clock
        )
    with pytest.raises(ValueError, match="precede"):
        LeasePool.from_checkpoint(checkpoint, clock=LogicalClock(1))


def test_concurrent_acquisitions_are_serialized_without_overlap() -> None:
    clock = LogicalClock()
    pool = LeasePool((0, 100), clock=clock)

    with ThreadPoolExecutor(max_workers=12) as executor:
        leases = tuple(
            executor.map(
                lambda owner: pool.acquire(owner, ttl=10),
                (f"owner-{index}" for index in range(100)),
            )
        )

    assert sorted(lease.resource.start for lease in leases) == list(range(100))
    assert sorted(lease.token for lease in leases) == list(range(1, 101))
    assert pool.diagnostics().available_capacity == 0
    with pytest.raises(LeaseUnavailableError):
        pool.acquire("overflow", ttl=1)
    assert pool.diagnostics().next_fencing_token == 101


def test_downstream_fence_validator_tracks_a_local_high_water_mark() -> None:
    validator = FenceValidator()

    assert validator.validate_fence("resource-a", 4)
    assert validator.validate_fence("resource-a", 4)  # idempotent retry
    assert not validator.validate_fence("resource-a", 3)
    assert validator.validate_fence("resource-a", 8)
    assert validator.validate_fence("resource-b", 1)
    assert validator.highest_token("resource-a") == 8
    with pytest.raises(ValueError, match="greater than zero"):
        validator.validate_fence("resource-a", 0)
    with pytest.raises(TypeError, match="hashable"):
        validator.validate_fence([], 1)  # type: ignore[arg-type]
