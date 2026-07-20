"""Database key-range lock contract."""

import pytest

from tests.oracles.applications.catalogs.database_key_range_locks import conflicts
from treemendous.applications._shared.locks import LockConflictError, LockMode
from treemendous.applications.catalogs.database_key_range_locks import (
    DatabaseKeyRangeLocks,
    KeyLockHandle,
    create_lock_table,
    encode_key,
)


def test_database_locks_use_stable_ordered_encoding_and_lock_modes() -> None:
    assert encode_key("a") < encode_key("aa") < encode_key("b")
    assert encode_key("alpha") == encode_key(b"alpha")
    locks = DatabaseKeyRangeLocks()
    handle = locks.acquire("users", "tx-1", "a", "m", LockMode.SHARED)
    rows = [("one", "users", "tx-1", encode_key("a"), encode_key("m"), "shared")]
    expected = conflicts(
        rows, "users", "tx-2", encode_key("c"), encode_key("d"), "exclusive"
    )
    actual = locks.conflicts("users", "tx-2", "c", "d", LockMode.EXCLUSIVE)
    assert len(actual) == len(expected) == 1
    with pytest.raises(LockConflictError):
        locks.acquire("users", "tx-2", "c", "d", LockMode.EXCLUSIVE)
    assert locks.upgrade(handle, owner="tx-1").mode is LockMode.EXCLUSIVE
    with pytest.raises(PermissionError):
        locks.release(handle, owner="tx-2")
    assert locks.release(handle, owner="tx-1").handle == handle
    assert not locks.snapshot().locks


@pytest.mark.parametrize(
    ("key", "error", "message"),
    [
        (1, TypeError, "str or bytes"),
        ("", ValueError, "must not be empty"),
        (b"x" * 33, ValueError, "exceeds 32 bytes"),
    ],
)
def test_key_encoding_rejects_ambiguous_or_unbounded_keys(
    key: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        encode_key(key)  # type: ignore[arg-type]


def test_database_lock_validators_and_unknown_table_handles_are_explicit() -> None:
    locks = create_lock_table()

    with pytest.raises(ValueError, match="start_key must sort before end_key"):
        locks.acquire("users", "tx", "z", "a", "shared")
    with pytest.raises(ValueError, match="start_key must sort before end_key"):
        locks.conflicts("users", "tx", "same", "same", "shared")
    with pytest.raises(ValueError, match="table must be a nonempty string"):
        locks.acquire("", "tx", "a", "b", "shared")
    with pytest.raises(ValueError, match="owner must be a nonempty string"):
        locks.acquire("users", "", "a", "b", "shared")

    handle = locks.acquire("users", "tx", "a", "b", "shared")
    unknown = KeyLockHandle("missing", handle.lock)
    with pytest.raises(KeyError, match="missing"):
        locks.upgrade(unknown, owner="tx")
    with pytest.raises(KeyError, match="missing"):
        locks.release(unknown, owner="tx")


def test_database_lock_snapshot_is_table_sorted_and_identity_preserving() -> None:
    locks = create_lock_table()
    later_table = locks.acquire("z_table", "tx-z", "b", "c", "exclusive")
    first_table_later_lock = locks.acquire("a_table", "tx-2", "c", "d", "shared")
    first_table_first_lock = locks.acquire("a_table", "tx-1", "a", "b", "shared")

    snapshot = locks.snapshot()
    actual_handles = tuple(lock.handle for lock in snapshot.locks)
    expected_handles = (
        first_table_later_lock,
        first_table_first_lock,
        later_table,
    )
    assert actual_handles == expected_handles

    actual_key_ranges = tuple((lock.start_key, lock.end_key) for lock in snapshot.locks)
    expected_key_ranges = (("c", "d"), ("a", "b"), ("b", "c"))
    assert actual_key_ranges == expected_key_ranges

    unseen_conflicts = locks.conflicts("unseen", "tx", "a", "b", "exclusive")
    assert not unseen_conflicts
