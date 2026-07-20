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


@pytest.mark.parametrize(
    ("cidr", "reserve_network", "reserve_broadcast", "expected"),
    [
        (
            "192.0.2.0/31",
            False,
            False,
            ("192.0.2.0", "192.0.2.1"),
        ),
        (
            "192.0.2.7/32",
            False,
            False,
            ("192.0.2.7",),
        ),
        ("2001:db8::/127", True, None, ("2001:db8::1",)),
        (
            "2001:db8::1/128",
            False,
            None,
            ("2001:db8::1",),
        ),
    ],
)
def test_address_oracle_matches_small_prefix_reservation_policy(
    cidr: str,
    reserve_network: bool,
    reserve_broadcast: bool | None,
    expected: tuple[str, ...],
) -> None:
    clock = LogicalClock()
    engine = NumericIPAddressPool(
        cidr,
        clock=clock,
        reserve_network=reserve_network,
        reserve_broadcast=reserve_broadcast,
    )
    oracle = NaiveAddressOracle(
        cidr,
        reserve_network=reserve_network,
        reserve_broadcast=reserve_broadcast,
    )

    _, oracle_block = oracle.acquire(len(expected), 2)
    lease = engine.acquire("owner", ttl=2, count=len(expected))

    assert tuple(str(ipaddress.ip_address(value)) for value in oracle_block) == expected
    assert (
        tuple(
            str(ipaddress.ip_address(value))
            for value in range(lease.resource.start, lease.resource.end)
        )
        == expected
    )


@pytest.mark.parametrize("cidr", ["192.0.2.0/31", "192.0.2.7/32", "2001:db8::1/128"])
def test_fully_reserved_small_prefix_has_no_allocatable_domain(cidr: str) -> None:
    oracle = NaiveAddressOracle(cidr)
    assert not oracle.free
    with pytest.raises(ValueError, match="complete leasing domain"):
        NumericIPAddressPool(cidr, clock=LogicalClock())


def test_address_oracle_matches_explicit_and_broadcast_reservations() -> None:
    reserved = ("192.0.2.2",)
    engine = NumericIPAddressPool(
        "192.0.2.0/29",
        clock=LogicalClock(),
        reserved=reserved,
        reserve_network=False,
        reserve_broadcast=False,
    )
    oracle = NaiveAddressOracle(
        "192.0.2.0/29",
        reserved=reserved,
        reserve_network=False,
        reserve_broadcast=False,
    )
    assert engine.diagnostics().pools[0][1].total_capacity == len(oracle.free) == 7

    ipv6 = NumericIPAddressPool(
        "2001:db8::/127",
        clock=LogicalClock(),
        reserve_network=False,
        reserve_broadcast=True,
    )
    ipv6_oracle = NaiveAddressOracle(
        "2001:db8::/127",
        reserve_network=False,
        reserve_broadcast=True,
    )
    assert len(ipv6_oracle.free) == 1
    assert str(ipv6.first_address(ipv6.acquire("owner", ttl=1))) == "2001:db8::"


def test_address_oracle_and_engine_reject_family_and_boundary_errors() -> None:
    engine = NumericIPAddressPool(
        "192.0.2.0/31",
        clock=LogicalClock(),
        reserve_network=False,
        reserve_broadcast=False,
    )
    oracle = NaiveAddressOracle(
        "192.0.2.0/31",
        reserve_network=False,
        reserve_broadcast=False,
    )

    with pytest.raises(ValueError, match="outside the CIDR"):
        engine.acquire("owner", ttl=1, count=2, start_address="192.0.2.1")
    with pytest.raises(ValueError, match="outside the CIDR"):
        oracle.acquire(2, 1, start_address="192.0.2.1")
    with pytest.raises(ValueError, match="inside"):
        NumericIPAddressPool(
            "192.0.2.0/31",
            clock=LogicalClock(),
            reserved=("2001:db8::1",),
            reserve_network=False,
            reserve_broadcast=False,
        )
    with pytest.raises(ValueError, match="inside"):
        NaiveAddressOracle(
            "192.0.2.0/31",
            reserved=("2001:db8::1",),
            reserve_network=False,
            reserve_broadcast=False,
        )
