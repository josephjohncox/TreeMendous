"""Database key-range locks using an explicit stable integer encoding."""

from __future__ import annotations

from dataclasses import dataclass

from treemendous.applications._shared.locks import (
    LockConflictError,
    LockHandle,
    LockMode,
    RangeLock,
    RangeLockTable,
)

_MAX_KEY_BYTES = 32
_BASE = 257


def encode_key(key: str | bytes) -> int:
    """Encode a bounded key to an order-preserving, stable integer.

    UTF-8 bytes are represented as ``byte + 1`` base-257 digits followed by
    zero padding. Consequently bytewise key ordering, including prefix order,
    equals integer ordering. Keys are limited to 32 encoded bytes so that
    padding has a fixed, explicit width.
    """
    if isinstance(key, str):
        raw = key.encode("utf-8")
    elif isinstance(key, bytes):
        raw = key
    else:
        raise TypeError("key must be str or bytes")
    if not raw:
        raise ValueError("key must not be empty")
    if len(raw) > _MAX_KEY_BYTES:
        raise ValueError("encoded key exceeds 32 bytes")
    result = 0
    for index in range(_MAX_KEY_BYTES):
        digit = raw[index] + 1 if index < len(raw) else 0
        result = result * _BASE + digit
    return result


@dataclass(frozen=True)
class KeyLockHandle:
    """Stable identity of one table-qualified key lock."""

    table: str
    lock: LockHandle[str]


@dataclass(frozen=True)
class KeyRangeLock:
    """One integer-encoded database key-range lock."""

    handle: KeyLockHandle
    start_key: str | bytes
    end_key: str | bytes
    encoded_start: int
    encoded_end: int
    mode: LockMode
    owner: str
    insertion_order: int


@dataclass(frozen=True)
class DatabaseLockSnapshot:
    """Deterministic active-lock snapshot."""

    locks: tuple[KeyRangeLock, ...]


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


class DatabaseKeyRangeLocks:
    """Shared/exclusive transaction locks over table-local encoded key bands."""

    def __init__(self) -> None:
        self._tables: dict[str, RangeLockTable[str]] = {}
        self._keys: dict[KeyLockHandle, tuple[str | bytes, str | bytes]] = {}

    def _table(self, table: str) -> RangeLockTable[str]:
        return self._tables.setdefault(_text(table, "table"), RangeLockTable())

    def _qualified(self, table: str, lock: RangeLock[str]) -> KeyRangeLock:
        handle = KeyLockHandle(table, lock.handle)
        start_key, end_key = self._keys[handle]
        return KeyRangeLock(
            handle,
            start_key,
            end_key,
            lock.start,
            lock.end,
            lock.mode,
            lock.owner,
            lock.insertion_order,
        )

    def acquire(
        self,
        table: str,
        owner: str,
        start_key: str | bytes,
        end_key: str | bytes,
        mode: LockMode | str,
    ) -> KeyLockHandle:
        """Acquire a transaction lock over ``[start_key, end_key)``."""
        encoded_start = encode_key(start_key)
        encoded_end = encode_key(end_key)
        if encoded_start >= encoded_end:
            raise ValueError("start_key must sort before end_key")
        table_name = _text(table, "table")
        lock = self._table(table_name).acquire(
            _text(owner, "owner"), encoded_start, encoded_end, mode
        )
        handle = KeyLockHandle(table_name, lock)
        self._keys[handle] = (start_key, end_key)
        return handle

    def conflicts(
        self,
        table: str,
        owner: str,
        start_key: str | bytes,
        end_key: str | bytes,
        mode: LockMode | str,
    ) -> tuple[KeyRangeLock, ...]:
        """Return exact incompatible table-local locks."""
        encoded_start = encode_key(start_key)
        encoded_end = encode_key(end_key)
        if encoded_start >= encoded_end:
            raise ValueError("start_key must sort before end_key")
        table_name = _text(table, "table")
        lock_table = self._tables.get(table_name)
        if lock_table is None:
            lock_table = RangeLockTable()
        return tuple(
            self._qualified(table_name, lock)
            for lock in lock_table.conflicts(owner, encoded_start, encoded_end, mode)
        )

    def upgrade(self, handle: KeyLockHandle, *, owner: str) -> KeyRangeLock:
        """Upgrade a shared lock while explicitly asserting transaction owner."""
        table = self._tables.get(handle.table)
        if table is None:
            raise KeyError(handle)
        return self._qualified(handle.table, table.upgrade(handle.lock, owner=owner))

    def release(self, handle: KeyLockHandle, *, owner: str) -> KeyRangeLock:
        """Release one lock identity with mandatory owner assertion."""
        table = self._tables.get(handle.table)
        if table is None:
            raise KeyError(handle)
        released = self._qualified(
            handle.table, table.release(handle.lock, owner=_text(owner, "owner"))
        )
        del self._keys[handle]
        return released

    def snapshot(self) -> DatabaseLockSnapshot:
        """Return table-name/acquisition-order active locks."""
        return DatabaseLockSnapshot(
            tuple(
                self._qualified(table_name, lock)
                for table_name in sorted(self._tables)
                for lock in self._tables[table_name].snapshot().locks
            )
        )


def create_lock_table() -> DatabaseKeyRangeLocks:
    """Create an empty database key-range lock engine."""
    return DatabaseKeyRangeLocks()


__all__ = [
    "DatabaseKeyRangeLocks",
    "DatabaseLockSnapshot",
    "KeyLockHandle",
    "KeyRangeLock",
    "LockConflictError",
    "LockMode",
    "create_lock_table",
    "encode_key",
]
