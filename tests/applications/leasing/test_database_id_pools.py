"""Focused contracts for monotonic and reusable database ID batches."""

from __future__ import annotations

import pytest

from tests.oracles.applications.leasing.database_id_pools import (
    NaiveDatabaseIdOracle,
)
from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.database_ids import (
    CommittedIdError,
    DatabaseIdPool,
)


def test_monotonic_commit_and_explicit_reuse_match_naive_model() -> None:
    clock = LogicalClock()
    engine = DatabaseIdPool("orders", maximum_id=20, clock=clock)
    oracle = NaiveDatabaseIdOracle(20)

    token, expected = oracle.acquire(3, 2)
    first = engine.acquire("writer-a", ttl=2, count=3, request_id="batch-a")
    assert first.token == token
    assert set(range(first.resource.start, first.resource.end)) == expected
    assert engine.acquire("writer-a", ttl=2, count=3, request_id="batch-a") == first
    committed = engine.commit(first)
    oracle.commit(token)
    assert set(range(committed.resource.start, committed.resource.end)) == oracle.committed
    with pytest.raises(CommittedIdError):
        engine.release(first)

    temporary = engine.acquire("writer-b", ttl=2, count=2)
    assert temporary.resource.start == 4
    clock.advance(2)
    assert engine.expire()
    recycled = engine.acquire("retry", ttl=3, count=2, reusable=True)
    assert recycled.resource == temporary.resource
    monotonic = engine.acquire("writer-c", ttl=3)
    assert monotonic.resource.start == 6


def test_database_fencing_snapshot_and_checkpoint_preserve_commit_boundary() -> None:
    clock = LogicalClock(5)
    engine = DatabaseIdPool("events", maximum_id=10, clock=clock)
    old = engine.acquire("old", ttl=2)
    assert engine.validate_fence(old, 1)
    engine.release(old)
    new = engine.acquire("new", ttl=2, reusable=True)
    assert engine.validate_fence(new, 1)
    assert not engine.validate_fence(old, 1)
    committed = engine.commit(new)

    snapshot = engine.snapshot()
    assert len(snapshot.committed) == 1
    assert snapshot.committed[0] == committed
    restored = DatabaseIdPool.from_checkpoint(engine.checkpoint(), clock=clock)
    restored_snapshot = restored.snapshot()
    assert restored_snapshot.next_monotonic_id == 2
    assert len(restored_snapshot.committed) == 1
    assert restored_snapshot.committed[0].resource == committed.resource
    assert restored.diagnostics().pools[0][1].released_leases == 2
