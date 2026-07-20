"""Focused contracts for scoped VLAN tag leasing."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.oracles.applications.leasing.vlan_tag_pools import NaiveVlanOracle
from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing._common import PoolGroup
from treemendous.applications.leasing.vlan_tags import VlanTagPool, VlanUnavailableError
from treemendous.domain import Span


def test_vlan_bounds_reservations_scopes_and_expiry_match_naive_model() -> None:
    clock = LogicalClock()
    engine = VlanTagPool(
        ("campus", "datacenter"),
        reserved_ranges=((1, 2),),
        scope_reserved={"campus": ((100, 110),)},
        clock=clock,
    )
    oracle = NaiveVlanOracle(("datacenter",), {1, 2})
    _, expected = oracle.acquire("datacenter", 2, 2)
    dc = engine.acquire("datacenter", "controller", ttl=2, count=2)
    campus = engine.acquire("campus", "controller", ttl=2, count=2)
    assert set(range(dc.resource.start, dc.resource.end)) == expected
    assert dc.resource == campus.resource
    with pytest.raises(ValueError, match="1..4094"):
        engine.acquire("campus", "bad", ttl=1, start_tag=0)
    with pytest.raises(ValueError, match="1..4094"):
        engine.acquire("campus", "bad", ttl=1, start_tag=4095)
    with pytest.raises(VlanUnavailableError):
        engine.acquire("campus", "reserved", ttl=1, start_tag=100)

    clock.advance(2)
    assert len(engine.expire()) == 2
    replacement = engine.acquire("datacenter", "next", ttl=2, count=2)
    assert replacement.resource == dc.resource


def test_vlan_idempotency_fencing_renew_release_and_checkpoint() -> None:
    clock = LogicalClock(5)
    engine = VlanTagPool(("edge",), clock=clock)
    old = engine.acquire("edge", "a", ttl=2, request_id="vlan-1")
    assert engine.acquire("edge", "a", ttl=2, request_id="vlan-1") == old
    assert engine.validate_fence(old, 1)
    renewed = engine.renew(old, ttl=3)
    engine.release(renewed)
    new = engine.acquire("edge", "b", ttl=2)
    assert engine.validate_fence(new, 1)
    assert not engine.validate_fence(old, 1)
    assert engine.snapshot().pools[0][1].diagnostics.active_leases == 1

    restored = VlanTagPool.from_checkpoint(engine.checkpoint(), clock=clock)
    diagnostics = restored.diagnostics().pools[0][1]
    assert diagnostics.released_leases == 1
    assert diagnostics.active_leases == 1


def test_identical_domains_cannot_swap_scoped_pool_lineages() -> None:
    engine = VlanTagPool(("one", "two"), clock=LogicalClock())
    checkpoint = engine.checkpoint()
    by_scope = {entry.scope: entry for entry in checkpoint.group.pools}

    with pytest.raises(ValueError, match="lineage does not match"):
        replace(by_scope["one"], pool=by_scope["two"].pool)


def test_invalid_scope_is_rejected_before_pool_acquisition_mutates_state() -> None:
    clock = LogicalClock()
    group = PoolGroup({"edge": (Span(1, 3),)}, clock=clock)

    with pytest.raises(TypeError, match="scope must be a string"):
        group.acquire(1, "owner", ttl=2)  # type: ignore[arg-type]

    snapshot = group.snapshot().pools[0][1]
    assert not snapshot.leases
    assert snapshot.diagnostics.next_fencing_token == 1
