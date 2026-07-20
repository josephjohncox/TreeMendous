"""Contracts for owner-aware shared/exclusive half-open range locks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from treemendous.applications._shared.locks import (
    LockConflictError,
    LockHandle,
    LockMode,
    RangeLockTable,
)


def test_shared_compatibility_exclusive_conflicts_and_half_open_edges() -> None:
    table = RangeLockTable[str]()
    first = table.acquire("a", 0, 10, LockMode.SHARED)
    second = table.acquire("b", 5, 15, "shared")
    edge = table.acquire("c", 15, 20, LockMode.EXCLUSIVE)

    with pytest.raises(LockConflictError) as raised:
        table.acquire("d", 7, 16, LockMode.EXCLUSIVE)

    assert [lock.handle for lock in raised.value.conflicts] == [first, second, edge]
    assert raised.value.request.start == 7
    assert raised.value.request.end == 16
    assert raised.value.request.mode is LockMode.EXCLUSIVE
    # [20, 21) touches but does not overlap c's [15, 20).
    assert table.acquire("d", 20, 21, "exclusive") == LockHandle("d", 1)


def test_shared_request_conflicts_only_with_other_owner_exclusive() -> None:
    table = RangeLockTable[str]()
    exclusive = table.acquire("a", 0, 5, "exclusive")
    table.acquire("b", 5, 10, "shared")

    conflicts = table.conflicts("c", 2, 8, "shared")
    assert [lock.handle for lock in conflicts] == [exclusive]


def test_same_owner_is_reentrant_but_duplicate_locks_release_separately() -> None:
    table = RangeLockTable[str]()
    shared = table.acquire("owner", 0, 10, "shared")
    exclusive = table.acquire("owner", 2, 8, "exclusive")
    duplicate = table.acquire("owner", 2, 8, "exclusive")

    assert [shared.sequence, exclusive.sequence, duplicate.sequence] == [1, 2, 3]
    table.release(exclusive)
    assert table.get(duplicate).mode is LockMode.EXCLUSIVE
    assert len(table) == 2
    with pytest.raises(KeyError):
        table.get(exclusive)


def test_upgrade_is_atomic_and_only_succeeds_without_other_owner() -> None:
    table = RangeLockTable[str]()
    target = table.acquire("a", 0, 10, "shared")
    blocker = table.acquire("b", 5, 6, "shared")
    own_lock = table.acquire("a", 2, 3, "shared")
    before = table.snapshot()

    with pytest.raises(LockConflictError) as raised:
        table.upgrade(target)
    assert [lock.handle for lock in raised.value.conflicts] == [blocker]
    assert table.snapshot() == before

    table.release(blocker)
    upgraded = table.upgrade(target)
    assert upgraded.handle == target
    assert upgraded.mode is LockMode.EXCLUSIVE
    assert upgraded.insertion_order == 0
    assert table.upgrade(target) == upgraded
    assert table.get(own_lock).mode is LockMode.SHARED


def test_conflict_and_validation_failures_do_not_consume_handles() -> None:
    table = RangeLockTable[str]()
    held = table.acquire("held", 0, 10, "exclusive")
    before = table.snapshot()

    with pytest.raises(LockConflictError):
        table.acquire("waiting", 1, 2, "shared")
    with pytest.raises(ValueError, match="start < end"):
        table.acquire("waiting", 2, 2, "shared")
    with pytest.raises(ValueError, match="mode"):
        table.acquire("waiting", 11, 12, "invalid")
    assert table.snapshot() == before

    table.release(held)
    assert table.acquire("waiting", 1, 2, "shared") == LockHandle("waiting", 1)


def test_snapshot_diagnostics_and_concurrent_duplicate_owners() -> None:
    table = RangeLockTable[str]()
    with ThreadPoolExecutor(max_workers=8) as executor:
        handles = list(
            executor.map(
                lambda _: table.acquire("owner", 0, 10, "exclusive"), range(100)
            )
        )

    assert sorted(handle.sequence for handle in handles) == list(range(1, 101))
    diagnostics = table.diagnostics()
    assert diagnostics.lock_count == 100
    assert diagnostics.owner_count == 1
    assert diagnostics.shared_count == 0
    assert diagnostics.exclusive_count == 100
    assert diagnostics.next_insertion_order == 100
    snapshot = table.snapshot()
    assert len(snapshot.next_sequences) == 1
    next_owner, next_sequence = snapshot.next_sequences[0]
    assert next_owner == "owner"
    assert next_sequence == 101
    assert [lock.insertion_order for lock in snapshot.locks] == list(range(100))


def test_missing_release_and_unhashable_owner_leave_state_unchanged() -> None:
    table = RangeLockTable[str]()
    before = table.snapshot()
    with pytest.raises(KeyError):
        table.release(LockHandle("missing", 1))
    with pytest.raises(TypeError, match="hashable"):
        table.acquire([], 0, 1, "shared")  # type: ignore[arg-type]
    assert table.snapshot() == before
