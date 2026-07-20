"""Filesystem byte-lock contract."""

import pytest

from tests.oracles.applications.catalogs.filesystem_byte_locks import conflicts
from treemendous.applications._shared.locks import LockConflictError, LockMode
from treemendous.applications.catalogs.filesystem_byte_locks import FilesystemByteLocks


def test_file_locks_are_owner_aware_identity_preserving_and_releasable() -> None:
    locks = FilesystemByteLocks()
    first = locks.acquire("data.bin", "reader-a", 0, 10, LockMode.SHARED)
    second = locks.acquire("data.bin", "reader-b", 5, 12, LockMode.SHARED)
    rows = [
        ("a", "data.bin", "reader-a", 0, 10, "shared"),
        ("b", "data.bin", "reader-b", 5, 12, "shared"),
    ]
    expected = conflicts(rows, "data.bin", "writer", "exclusive", 8, 9)
    actual = locks.conflicts("data.bin", "writer", 8, 9, LockMode.EXCLUSIVE)
    assert len(actual) == len(expected) == 2
    with pytest.raises(LockConflictError):
        locks.acquire("data.bin", "writer", 8, 9, LockMode.EXCLUSIVE)
    with pytest.raises(PermissionError):
        locks.release(first, owner="reader-b")
    assert locks.release(first, owner="reader-a").handle == first
    assert locks.upgrade(second, owner="reader-b").mode is LockMode.EXCLUSIVE
    assert locks.snapshot().locks[0].handle == second
