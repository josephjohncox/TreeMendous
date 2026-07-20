"""Focused contracts for TCP/UDP port leasing."""

from __future__ import annotations

import pytest

from tests.oracles.applications.leasing.tcp_udp_port_leases import NaivePortOracle
from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.tcp_udp_ports import (
    PortLeaseEngine,
    PortUnavailableError,
)


def test_ports_match_naive_oracle_and_keep_protocol_namespaces_separate() -> None:
    clock = LogicalClock()
    engine = PortLeaseEngine(clock=clock)
    oracle = NaivePortOracle()

    _, expected = oracle.acquire("tcp", 3, 2)
    tcp = engine.acquire("tcp", "svc", ttl=2, count=3, request_id="tcp-1")
    udp = engine.acquire("udp", "dns", ttl=2, count=3, request_id="udp-1")

    assert tuple(range(tcp.resource.start, tcp.resource.end)) == expected
    assert udp.resource == tcp.resource
    assert engine.acquire("tcp", "svc", ttl=2, count=3, request_id="tcp-1") == tcp
    with pytest.raises(PortUnavailableError):
        engine.acquire("tcp", "rootless", ttl=2, start_port=22)
    with pytest.raises(PortUnavailableError):
        engine.acquire("udp", "bad", ttl=2, start_port=50000)

    assert engine.validate_fence(tcp, tcp.resource.start)
    clock.advance(2)
    assert len(engine.expire()) == 2
    replacement = engine.acquire("tcp", "next", ttl=2, count=3)
    assert replacement.resource == tcp.resource
    assert replacement.token > tcp.token
    assert engine.validate_fence(replacement, replacement.resource.start)
    assert not engine.validate_fence(tcp, tcp.resource.start)


def test_port_renew_release_snapshot_checkpoint_and_restore() -> None:
    clock = LogicalClock(10)
    engine = PortLeaseEngine(clock=clock, ephemeral_ports=None)
    original = engine.acquire("tcp", "api", ttl=3, count=2)
    renewed = engine.renew(original, ttl=5)
    assert renewed.revision == 2
    released = engine.release(renewed)
    assert not released.lease.active
    assert engine.snapshot().pools[0][1].diagnostics.released_leases == 1
    restored = PortLeaseEngine.from_checkpoint(engine.checkpoint(), clock=clock)
    assert restored.diagnostics().pools[0][1].released_leases == 1
    original_ids = {scope: pool.pool_id for scope, pool in engine.snapshot().pools}
    restored_ids = {scope: pool.pool_id for scope, pool in restored.snapshot().pools}
    assert restored_ids["tcp"] != original_ids["tcp"]
