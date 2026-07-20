"""Focused contracts for shard region ownership and handoff fencing."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.oracles.applications.leasing.game_world_region_ids import NaiveRegionOracle
from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications._shared.leasing import (
    LeaseRequestConflictError,
    LeaseState,
)
from treemendous.applications.leasing._common import NumericLease
from treemendous.applications.leasing.game_regions import (
    GameRegionPool,
    RegionAdjacencyError,
)


class _SequenceClock:
    def __init__(self, values: list[int]) -> None:
        self._values = iter(values)
        self.calls = 0

    def now(self) -> int:
        self.calls += 1
        return next(self._values)


def test_shard_adjacency_and_handoff_match_naive_ownership_model() -> None:
    clock = LogicalClock()
    engine = GameRegionPool({"west": (10, 20)}, clock=clock)
    oracle = NaiveRegionOracle({"west": set(range(10, 21))})
    token, expected = oracle.acquire("west", "server-a", 2, 4)
    first = engine.acquire("west", "server-a", ttl=4, count=2, request_id="claim")
    assert first.token == token
    assert set(range(first.resource.start, first.resource.end)) == expected
    adjacent = engine.acquire(
        "west",
        "server-a",
        ttl=4,
        count=2,
        adjacent_to=first,
        request_id="expand",
    )
    assert adjacent.resource.start == first.resource.end
    assert (
        engine.acquire(
            "west",
            "server-a",
            ttl=4,
            count=2,
            adjacent_to=first,
            request_id="expand",
        )
        == adjacent
    )
    with pytest.raises(RegionAdjacencyError):
        engine.acquire("west", "other", ttl=2, adjacent_to=first)

    oracle_token, oracle_values = oracle.handoff(token, "server-b", 5)
    transferred = engine.handoff(
        first,
        "server-b",
        ttl=5,
        request_id="handoff-1",
    )
    assert transferred.token == oracle_token + 1  # adjacency consumed one token
    assert (
        set(range(transferred.resource.start, transferred.resource.end))
        == oracle_values
    )
    assert (
        engine.handoff(first, "server-b", ttl=5, request_id="handoff-1") == transferred
    )
    assert engine.validate_fence(first, first.resource.start)
    assert engine.validate_fence(transferred, transferred.resource.start)
    assert not engine.validate_fence(first, first.resource.start)

    restored = GameRegionPool.from_checkpoint(engine.checkpoint(), clock=clock)
    restored_transfer = restored.handoff(
        first,
        "server-b",
        ttl=5,
        request_id="handoff-1",
    )
    assert restored_transfer.token == transferred.token
    assert restored_transfer.pool_id != transferred.pool_id


def test_region_renew_release_expiry_snapshot_and_checkpoint() -> None:
    clock = LogicalClock(3)
    engine = GameRegionPool({"one": (1, 4), "two": (1, 4)}, clock=clock)
    lease = engine.acquire("one", "server", ttl=2)
    renewed = engine.renew(lease, ttl=3)
    engine.release(renewed)
    elapsed = engine.acquire("two", "server", ttl=1)
    clock.advance()
    expired = engine.expire()
    assert len(expired) == 1
    assert expired[0].token == elapsed.token
    assert expired[0].resource == elapsed.resource
    assert len(engine.snapshot().pools) == 2

    restored = GameRegionPool.from_checkpoint(engine.checkpoint(), clock=clock)
    diagnostics = dict(restored.diagnostics().pools)
    assert diagnostics["one"].released_leases == 1
    assert diagnostics["two"].expired_leases == 1


def test_region_request_fingerprints_mode_start_and_stable_anchor_identity() -> None:
    clock = LogicalClock()
    engine = GameRegionPool({"west": (10, 30)}, clock=clock)
    first_anchor = engine.acquire("west", "server", ttl=10, start_region=10)
    other_anchor = engine.acquire("west", "server", ttl=10, start_region=20)
    adjacent = engine.acquire(
        "west",
        "server",
        ttl=5,
        adjacent_to=first_anchor,
        request_id="adjacent",
    )

    with pytest.raises(LeaseRequestConflictError):
        engine.acquire(
            "west",
            "server",
            ttl=5,
            adjacent_to=other_anchor,
            request_id="adjacent",
        )

    engine.release(first_anchor)
    engine.release(adjacent)
    terminal_retry = engine.acquire(
        "west",
        "server",
        ttl=5,
        adjacent_to=first_anchor,
        request_id="adjacent",
    )
    assert terminal_retry.token == adjacent.token
    assert terminal_retry.lease.state is LeaseState.RELEASED

    restored = GameRegionPool.from_checkpoint(engine.checkpoint(), clock=clock)
    restored_retry = restored.acquire(
        "west",
        "server",
        ttl=5,
        adjacent_to=first_anchor,
        request_id="adjacent",
    )
    assert restored_retry.token == adjacent.token
    assert restored_retry.pool_id != adjacent.pool_id
    assert restored_retry.lease.state is LeaseState.RELEASED

    automatic = engine.acquire("west", "server", ttl=5, request_id="selection")
    with pytest.raises(LeaseRequestConflictError):
        engine.acquire(
            "west",
            "server",
            ttl=5,
            start_region=automatic.resource.start,
            request_id="selection",
        )


def test_adjacent_request_replay_rejects_foreign_anchor_lineage() -> None:
    clock = LogicalClock()
    engine = GameRegionPool({"west": (1, 4)}, clock=clock)
    anchor = engine.acquire("west", "server", ttl=5)
    engine.acquire(
        "west",
        "server",
        ttl=5,
        adjacent_to=anchor,
        request_id="adjacent",
    )
    foreign_anchor = NumericLease(
        "west",
        replace(anchor.lease, pool_id="foreign-lineage"),
    )

    with pytest.raises(LeaseRequestConflictError):
        engine.acquire(
            "west",
            "server",
            ttl=5,
            adjacent_to=foreign_anchor,
            request_id="adjacent",
        )


def test_region_handoff_uses_one_clock_observation_and_is_failure_atomic() -> None:
    clock = _SequenceClock([0, 1, 2, 2])
    engine = GameRegionPool({"west": (1, 2)}, clock=clock)
    source = engine.acquire("west", "source", ttl=5)
    calls_before = clock.calls

    transferred = engine.handoff(
        source,
        "target",
        ttl=5,
        request_id="handoff",
    )

    assert clock.calls == calls_before + 1
    assert transferred.resource == source.resource
    assert (
        engine.handoff(
            source,
            "target",
            ttl=5,
            request_id="handoff",
        )
        == transferred
    )

    regressing = _SequenceClock([0, 1, 0, 1])
    failed = GameRegionPool({"west": (1, 1)}, clock=regressing)
    still_active = failed.acquire("west", "source", ttl=5)
    with pytest.raises(RuntimeError, match="backwards"):
        failed.handoff(
            still_active,
            "target",
            ttl=5,
            request_id="failed-handoff",
        )
    snapshot = failed.snapshot().pools[0][1]
    assert len(snapshot.leases) == 1
    assert snapshot.leases[0] == still_active.lease
    assert snapshot.leases[0].state is LeaseState.ACTIVE
    assert snapshot.diagnostics.next_fencing_token == 2
