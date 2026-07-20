"""Database key-range lock contract."""

import pytest

from tests.oracles.applications.catalogs.database_key_range_locks import conflicts
from treemendous.applications._shared.locks import LockConflictError, LockMode
from treemendous.applications.catalogs.database_key_range_locks import (
    DatabaseKeyRangeLocks,
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
