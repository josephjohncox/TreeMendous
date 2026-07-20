"""Focused contracts for reserved phone extension plans."""

from __future__ import annotations

import pytest

from tests.oracles.applications.leasing.phone_extension_pools import (
    NaiveExtensionOracle,
)
from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.phone_extensions import (
    ExtensionUnavailableError,
    PhoneExtensionPool,
)


def test_numbering_plan_reservations_and_expiry_match_naive_model() -> None:
    clock = LogicalClock()
    reserved = {101, 105, 106}
    engine = PhoneExtensionPool(
        "hq",
        first_extension=100,
        last_extension=110,
        emergency_numbers=(101,),
        service_ranges=((105, 106),),
        clock=clock,
    )
    oracle = NaiveExtensionOracle(100, 110, reserved)
    _, expected = oracle.acquire(2, 2)
    lease = engine.acquire("provisioner", ttl=2, count=2, request_id="user-1")
    assert set(range(lease.resource.start, lease.resource.end)) == expected
    assert engine.acquire("provisioner", ttl=2, count=2, request_id="user-1") == lease
    with pytest.raises(ExtensionUnavailableError):
        engine.acquire("bad", ttl=1, start_extension=101)
    with pytest.raises(ExtensionUnavailableError):
        engine.acquire("bad", ttl=1, count=2, start_extension=105)

    clock.advance(2)
    assert len(engine.expire()) == 1
    replacement = engine.acquire("next", ttl=2, count=2)
    assert replacement.resource == lease.resource


def test_extension_fencing_renew_release_snapshot_and_checkpoint() -> None:
    clock = LogicalClock(4)
    engine = PhoneExtensionPool(
        "branch",
        first_extension=200,
        last_extension=202,
        emergency_numbers=(),
        clock=clock,
    )
    old = engine.acquire("pbx-a", ttl=2)
    assert engine.validate_fence(old, 200)
    renewed = engine.renew(old, ttl=3)
    assert renewed.revision == 2
    engine.release(renewed)
    new = engine.acquire("pbx-b", ttl=2)
    assert engine.validate_fence(new, 200)
    assert not engine.validate_fence(old, 200)
    assert engine.snapshot().pools[0][1].diagnostics.active_leases == 1

    restored = PhoneExtensionPool.from_checkpoint(engine.checkpoint(), clock=clock)
    assert restored.plan_id == "branch"
    diagnostics = restored.diagnostics().pools[0][1]
    assert diagnostics.released_leases == 1
    assert diagnostics.active_leases == 1
