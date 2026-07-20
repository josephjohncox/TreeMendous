"""Focused contracts for shard region ownership and handoff fencing."""

from __future__ import annotations

import pytest

from tests.oracles.applications.leasing.game_world_region_ids import NaiveRegionOracle
from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.game_regions import (
    GameRegionPool,
    RegionAdjacencyError,
)


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
    assert engine.acquire(
        "west",
        "server-a",
        ttl=4,
        count=2,
        adjacent_to=first,
        request_id="expand",
    ) == adjacent
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
    assert set(range(transferred.resource.start, transferred.resource.end)) == oracle_values
    assert engine.handoff(first, "server-b", ttl=5, request_id="handoff-1") == transferred
    assert engine.validate_fence(first, first.resource.start)
    assert engine.validate_fence(transferred, transferred.resource.start)
    assert not engine.validate_fence(first, first.resource.start)


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
