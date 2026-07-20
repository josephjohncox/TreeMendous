"""Focused contracts for integer-encoded CIDR address pools."""

from __future__ import annotations

import ipaddress

import pytest

from tests.oracles.applications.leasing.numeric_ip_address_pools import (
    NaiveAddressOracle,
)
from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.numeric_ip_pools import (
    AddressUnavailableError,
    NumericIPAddressPool,
)


def test_address_pool_matches_naive_hosts_and_enforces_cidr_reservations() -> None:
    clock = LogicalClock()
    engine = NumericIPAddressPool("192.0.2.0/29", clock=clock)
    oracle = NaiveAddressOracle("192.0.2.0/29")

    _, expected = oracle.acquire(2, 2)
    lease = engine.acquire("dhcp", ttl=2, count=2, request_id="discover-1")
    assert tuple(range(lease.resource.start, lease.resource.end)) == expected
    assert str(engine.first_address(lease)) == "192.0.2.1"
    assert str(engine.last_address(lease)) == "192.0.2.2"
    assert engine.acquire("dhcp", ttl=2, count=2, request_id="discover-1") == lease
    with pytest.raises(AddressUnavailableError):
        engine.acquire("bad", ttl=1, start_address="192.0.2.0")
    with pytest.raises(AddressUnavailableError):
        engine.acquire("bad", ttl=1, start_address="192.0.2.7")
    with pytest.raises(ValueError, match="inside"):
        engine.acquire("bad", ttl=1, start_address="192.0.3.1")

    assert engine.validate_fence(lease, ipaddress.ip_address("192.0.2.1"))
    clock.advance(2)
    engine.expire()
    replacement = engine.acquire("next", ttl=2, count=2)
    assert replacement.resource == lease.resource
    assert engine.validate_fence(replacement, "192.0.2.1")
    assert not engine.validate_fence(lease, "192.0.2.1")


def test_address_renew_release_checkpoint_and_diagnostics() -> None:
    clock = LogicalClock(4)
    engine = NumericIPAddressPool(
        "2001:db8::/125",
        clock=clock,
        reserved=("2001:db8::2",),
    )
    lease = engine.acquire("router", ttl=2)
    renewed = engine.renew(lease, ttl=4)
    engine.release(renewed)
    checkpoint = engine.checkpoint()
    restored = NumericIPAddressPool.from_checkpoint(checkpoint, clock=clock)
    assert restored.network == ipaddress.ip_network("2001:db8::/125")
    diagnostics = restored.diagnostics().pools[0][1]
    assert diagnostics.released_leases == 1
    assert restored.snapshot().pools[0][1].available_spans
