"""Publication, cache, and delta-staging contracts for the lease kernel."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications._shared.leasing import (
    InvalidLeaseError,
    Lease,
    LeasePool,
    LeaseState,
    LeaseUnavailableError,
)
from treemendous.applications.leasing._common import NumericLease, PoolGroup
from treemendous.domain import Span


def _cache_state(pool: LeasePool) -> tuple[object, ...]:
    return (
        pool._leases,
        pool._free,
        pool._lease_projection_cache,
        pool._available_projection_cache,
        pool._state_counts_cache,
        pool._next_fencing_token,
    )


def test_snapshot_projections_are_lazy_reused_and_point_in_time() -> None:
    clock = LogicalClock()
    pool = LeasePool((0, 8), clock=clock)
    first_lease = pool.acquire("first", ttl=10, size=2)
    assert pool._lease_projection_cache is None
    assert pool._available_projection_cache is None
    assert pool._state_counts_cache is None

    first = pool.snapshot()
    assert getattr(pool, "_lease_projection_cache") is first.leases
    assert getattr(pool, "_available_projection_cache") is first.available_spans
    counts = getattr(pool, "_state_counts_cache")
    second = pool.snapshot()

    assert second is not first
    assert second.leases is first.leases
    assert second.available_spans is first.available_spans
    assert getattr(pool, "_state_counts_cache") is counts
    assert second.diagnostics == first.diagnostics

    renewed = pool.renew(first_lease, ttl=12)
    assert getattr(pool, "_lease_projection_cache") is None
    assert getattr(pool, "_state_counts_cache") is None
    assert getattr(pool, "_available_projection_cache") is first.available_spans
    after_renew = pool.snapshot()
    assert after_renew.available_spans is first.available_spans
    assert after_renew.leases != first.leases

    pool.release(renewed)
    after_release = pool.snapshot()
    assert after_release.available_spans != first.available_spans
    assert first.leases == (first_lease,)
    assert first.available_spans == (Span(2, 8),)
    assert first.diagnostics.active_leases == 1


def test_failed_mutation_retains_published_state_and_all_projection_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = LeasePool((0, 2), clock=LogicalClock())
    pool.acquire("owner", ttl=10)
    pool.snapshot()
    before = _cache_state(pool)

    with pytest.raises(LeaseUnavailableError):
        pool.acquire("blocked", ttl=10, exact_span=(0, 1))
    assert _cache_state(pool) == before
    assert all(
        current is previous
        for current, previous in zip(_cache_state(pool)[:5], before[:5], strict=True)
    )

    def fail_selection(*args: object, **kwargs: object) -> Span:
        del args, kwargs
        raise RuntimeError("injected selection failure")

    monkeypatch.setattr(pool, "_select_span", fail_selection)
    with pytest.raises(RuntimeError, match="injected"):
        pool.acquire("next", ttl=10)
    after = _cache_state(pool)
    assert all(
        current is previous
        for current, previous in zip(after[:5], before[:5], strict=True)
    )
    assert after[5] == before[5]


def test_rejected_free_delta_candidate_is_not_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = LeasePool((0, 20), clock=LogicalClock())
    pool.acquire("owner", ttl=100)
    replays = 0
    original = pool._build_free

    def observed_rebuild(leases: dict[int, Lease]) -> object:
        nonlocal replays
        replays += len(leases)
        return original(leases)

    monkeypatch.setattr(pool, "_build_free", observed_rebuild)
    acquired = pool.acquire("new", ttl=100)
    pool.release(acquired)

    assert replays > 0
    assert not hasattr(pool, "_clone_free")


def test_fence_lookup_uses_no_public_snapshot_or_projection_scan_and_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = LogicalClock()
    group = PoolGroup({"scope": (Span(0, 4),)}, clock=clock)
    handle = group.acquire("scope", "owner", ttl=2)
    pool = group.pool("scope")

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("public projection path was used")

    monkeypatch.setattr(pool, "snapshot", forbidden)
    monkeypatch.setattr(pool, "_lease_projection", forbidden)
    assert group.validate_fence("key", handle)

    clock.advance(2)
    assert group.validate_fence("key", handle)
    assert pool._leases[handle.token].state is LeaseState.EXPIRED
    assert pool._free.intervals()[0].span == Span(0, 4)

    unknown = NumericLease("scope", replace(handle.lease, fencing_token=99))
    with pytest.raises(InvalidLeaseError, match="not issued"):
        group.validate_fence("other", unknown)


def test_expiration_boundary_and_checkpoint_restore_keep_indexes_consistent() -> None:
    clock = LogicalClock(5)
    pool = LeasePool((0, 4), clock=clock)
    first = pool.acquire("first", ttl=2)
    second = pool.acquire("second", ttl=3)
    clock.advance(1)
    assert pool.expire() == ()
    clock.advance(1)
    assert pool.expire() == (replace(first, state=LeaseState.EXPIRED),)
    assert pool.snapshot().leases[1] == second

    restored = LeasePool.from_checkpoint(pool.checkpoint(), clock=clock)
    assert restored.snapshot().leases[0].state is LeaseState.EXPIRED


def test_concurrent_snapshots_are_complete_and_share_only_immutable_projections() -> (
    None
):
    pool = LeasePool((0, 128), clock=LogicalClock())
    for index in range(64):
        pool.acquire(f"owner-{index}", ttl=100)

    def reader() -> tuple[object, object, int, int]:
        snapshot = pool.snapshot()
        return (
            snapshot.leases,
            snapshot.available_spans,
            len(snapshot.leases),
            snapshot.diagnostics.active_leases,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        observed = tuple(executor.map(lambda _: reader(), range(128)))

    assert {item[2:] for item in observed} == {(64, 64)}
    assert len({id(item[0]) for item in observed}) == 1
    assert len({id(item[1]) for item in observed}) == 1
