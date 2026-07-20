"""Focused contracts for compatible warehouse bin range leases."""

from __future__ import annotations

import pytest

from tests.oracles.applications.leasing.warehouse_bin_ranges import NaiveBinOracle
from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.warehouse_bins import (
    BinCompatibilityError,
    BinRequestConflictError,
    BinZone,
    WarehouseBinPool,
)


def test_bin_zone_size_hazard_and_expiry_match_naive_model() -> None:
    clock = LogicalClock()
    zone = BinZone(
        10,
        15,
        size_classes=frozenset({"small", "large"}),
        hazards=frozenset({"general", "flammable"}),
    )
    engine = WarehouseBinPool({"north": zone}, clock=clock)
    oracle = NaiveBinOracle(
        {"north": (set(range(10, 16)), {"small", "large"}, {"general", "flammable"})}
    )
    _, expected = oracle.acquire("north", 2, "large", "flammable", 2)
    lease = engine.acquire(
        "north",
        "putaway",
        ttl=2,
        count=2,
        size_class="large",
        hazard="flammable",
        request_id="job-1",
    )
    assert set(range(lease.resource.start, lease.resource.end)) == expected
    assert lease.size_class == "large"
    assert lease.hazard == "flammable"
    assert (
        engine.acquire(
            "north",
            "putaway",
            ttl=2,
            count=2,
            size_class="large",
            hazard="flammable",
            request_id="job-1",
        )
        == lease
    )
    with pytest.raises(BinCompatibilityError):
        engine.acquire("north", "bad", ttl=1, size_class="oversize")
    with pytest.raises(BinRequestConflictError):
        engine.acquire(
            "north",
            "putaway",
            ttl=2,
            count=2,
            size_class="small",
            request_id="job-1",
        )

    clock.advance(2)
    expired = engine.expire()
    assert len(expired) == 1
    replacement = engine.acquire("north", "next", ttl=2, count=2, size_class="small")
    assert replacement.resource == lease.resource


def test_bin_fencing_multiple_zones_snapshot_and_checkpoint() -> None:
    clock = LogicalClock(2)
    engine = WarehouseBinPool(
        {"A": BinZone(1, 2), "B": BinZone(1, 2)},
        clock=clock,
    )
    a = engine.acquire("A", "one", ttl=3)
    b = engine.acquire("B", "two", ttl=3)
    assert a.token == b.token == 1
    assert engine.validate_fence(a, 1)
    assert engine.validate_fence(b, 1)
    released = engine.release(a)
    assert released.inner.lease.state.value == "released"
    assert len(engine.snapshot().leases) == 2

    restored = WarehouseBinPool.from_checkpoint(engine.checkpoint(), clock=clock)
    snapshot = restored.snapshot()
    assert len(snapshot.leases) == 2
    assert {lease.inner.scope for lease in snapshot.leases} == {"A", "B"}
    assert restored.diagnostics().pools[0][1].released_leases == 1
