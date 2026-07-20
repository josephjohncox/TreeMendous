"""Focused contracts for entitled software seat checkouts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.oracles.applications.leasing.software_license_seats import NaiveSeatOracle
from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.software_seats import (
    EntitlementError,
    SoftwareSeatPool,
)


def test_product_entitlements_checkout_renewal_and_expiry_match_oracle() -> None:
    clock = LogicalClock()
    engine = SoftwareSeatPool(
        {"ide": 3, "render": 2},
        entitlements={"alice": {"ide": 2}, "bob": {"render": 1}},
        clock=clock,
    )
    oracle = NaiveSeatOracle(3, {"alice": 2})
    _, expected = oracle.checkout("alice", 2, 2)
    lease = engine.checkout("ide", "alice", ttl=2, count=2, request_id="login")
    assert set(range(lease.resource.start, lease.resource.end)) == expected
    assert engine.checkout("ide", "alice", ttl=2, count=2, request_id="login") == lease
    with pytest.raises(EntitlementError):
        engine.checkout("ide", "alice", ttl=2)
    with pytest.raises(EntitlementError):
        engine.checkout("ide", "mallory", ttl=2)

    clock.advance()
    renewed = engine.renew(lease, ttl=2)
    assert renewed.expires_at == 3
    clock.advance(2)
    assert len(engine.expire()) == 1
    replacement = engine.checkout("ide", "alice", ttl=2, count=2)
    assert replacement.resource == lease.resource


def test_seat_fencing_release_snapshot_and_entitlement_checkpoint() -> None:
    clock = LogicalClock(3)
    engine = SoftwareSeatPool(
        {"compiler": 1},
        entitlements={"team": {"compiler": 1}},
        clock=clock,
    )
    old = engine.checkout("compiler", "team", ttl=4)
    assert engine.validate_fence(old, 1)
    engine.release(old)
    new = engine.checkout("compiler", "team", ttl=4)
    assert engine.validate_fence(new, 1)
    assert not engine.validate_fence(old, 1)
    assert engine.snapshot().pools[0][1].diagnostics.active_leases == 1

    restored = SoftwareSeatPool.from_checkpoint(engine.checkpoint(), clock=clock)
    with pytest.raises(EntitlementError):
        restored.checkout("compiler", "other", ttl=2)
    assert restored.diagnostics().pools[0][1].active_leases == 1


def test_entitlement_policy_is_private_copy_and_renewal_uses_fixed_grant() -> None:
    clock = LogicalClock()
    grants = {"alice": {"ide": 1}}
    engine = SoftwareSeatPool(
        {"ide": 2},
        entitlements=grants,
        clock=clock,
    )
    lease = engine.checkout("ide", "alice", ttl=2, request_id="login")

    grants["alice"]["ide"] = 2
    with pytest.raises(EntitlementError):
        engine.checkout("ide", "alice", ttl=2)
    grants["alice"]["ide"] = 0
    grants["alice"].clear()

    assert engine.checkout("ide", "alice", ttl=2, request_id="login") == lease
    renewed = engine.renew(lease, ttl=5)
    assert renewed.expires_at == 5


def test_restore_rejects_policy_reduction_below_active_use() -> None:
    clock = LogicalClock()
    engine = SoftwareSeatPool(
        {"ide": 2},
        entitlements={"alice": {"ide": 2}},
        clock=clock,
    )
    engine.checkout("ide", "alice", ttl=5, count=2)
    checkpoint = engine.checkpoint()
    reduced = replace(
        checkpoint,
        entitlements=(("alice", (("ide", 1),)),),
    )

    with pytest.raises(ValueError, match="exceed checkpoint entitlement"):
        SoftwareSeatPool.from_checkpoint(reduced, clock=clock)

    removed = replace(checkpoint, entitlements=())
    with pytest.raises(ValueError, match="lack a checkpoint entitlement"):
        SoftwareSeatPool.from_checkpoint(removed, clock=clock)
