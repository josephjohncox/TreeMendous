"""Owner-aware shared/exclusive filesystem byte-range locks."""

from __future__ import annotations

from dataclasses import dataclass

from treemendous.applications._shared.locks import (
    LockConflictError,
    LockHandle,
    LockMode,
    RangeLock,
    RangeLockTable,
)
from treemendous.domain import Span


@dataclass(frozen=True)
class FileLockHandle:
    """Stable identity of a lock scoped to one file."""

    file: str
    lock: LockHandle[str]


@dataclass(frozen=True)
class FileLock:
    """One file-qualified byte lock."""

    handle: FileLockHandle
    start: int
    end: int
    mode: LockMode
    owner: str
    insertion_order: int


@dataclass(frozen=True)
class FilesystemLockSnapshot:
    """Deterministically ordered active file locks."""

    locks: tuple[FileLock, ...]


def _name(value: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


class FilesystemByteLocks:
    """Independent lock tables per file with mandatory owner release."""

    def __init__(self) -> None:
        self._tables: dict[str, RangeLockTable[str]] = {}

    def _table(self, file: str) -> RangeLockTable[str]:
        return self._tables.setdefault(_name(file, "file"), RangeLockTable())

    @staticmethod
    def _qualified(file: str, lock: RangeLock[str]) -> FileLock:
        return FileLock(
            FileLockHandle(file, lock.handle),
            lock.start,
            lock.end,
            lock.mode,
            lock.owner,
            lock.insertion_order,
        )

    def acquire(
        self,
        file: str,
        owner: str,
        start: int,
        end: int,
        mode: LockMode | str,
    ) -> FileLockHandle:
        """Acquire one lock; incompatible other-owner locks raise a conflict."""
        _name(owner, "owner")
        table = self._table(file)
        return FileLockHandle(file, table.acquire(owner, start, end, mode))

    def conflicts(
        self,
        file: str,
        owner: str,
        start: int,
        end: int,
        mode: LockMode | str,
    ) -> tuple[FileLock, ...]:
        """Return exact incompatible locks for a proposed acquisition."""
        table = self._tables.get(_name(file, "file"))
        if table is None:
            # Validate request through a temporary table without retaining it.
            table = RangeLockTable()
        return tuple(
            self._qualified(file, lock)
            for lock in table.conflicts(_name(owner, "owner"), start, end, mode)
        )

    def upgrade(self, handle: FileLockHandle, *, owner: str) -> FileLock:
        """Upgrade a shared lock while explicitly asserting its owner."""
        table = self._tables.get(handle.file)
        if table is None:
            raise KeyError(handle)
        return self._qualified(
            handle.file, table.upgrade(handle.lock, owner=_name(owner, "owner"))
        )

    def release(self, handle: FileLockHandle, *, owner: str) -> FileLock:
        """Release exactly one identity; the owning principal is mandatory."""
        table = self._tables.get(handle.file)
        if table is None:
            raise KeyError(handle)
        released = table.release(handle.lock, owner=_name(owner, "owner"))
        return self._qualified(handle.file, released)

    def active(self, file: str, start: int, end: int) -> tuple[FileLock, ...]:
        """Return every active lock intersecting a byte range."""
        Span(start, end)
        table = self._tables.get(_name(file, "file"))
        if table is None:
            return ()
        return tuple(
            self._qualified(file, lock)
            for lock in table.snapshot().locks
            if lock.start < end and start < lock.end
        )

    def snapshot(self) -> FilesystemLockSnapshot:
        """Return a stable file-name/acquisition-order snapshot."""
        locks = tuple(
            self._qualified(file, lock)
            for file in sorted(self._tables)
            for lock in self._tables[file].snapshot().locks
        )
        return FilesystemLockSnapshot(locks)


def create_lock_table() -> FilesystemByteLocks:
    """Create an empty filesystem byte-lock engine."""
    return FilesystemByteLocks()


__all__ = [
    "FileLock",
    "FileLockHandle",
    "FilesystemByteLocks",
    "FilesystemLockSnapshot",
    "LockConflictError",
    "LockMode",
    "create_lock_table",
]
