"""Owner-aware shared/exclusive half-open range lock kernel."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Generic, TypeVar
from uuid import UUID, uuid4

from treemendous.domain import Span, validate_coordinate

LockOwnerT = TypeVar("LockOwnerT", bound=Hashable)


class LockMode(str, Enum):
    """Compatibility mode for a range lock."""

    SHARED = "shared"
    EXCLUSIVE = "exclusive"


@dataclass(frozen=True)
class LockHandle(Generic[LockOwnerT]):
    """Table-scoped value identity, not authorization for a lock."""

    owner: LockOwnerT
    sequence: int
    lineage: UUID

    def __post_init__(self) -> None:
        hash(self.owner)
        validate_coordinate(self.sequence, "sequence")
        if self.sequence < 1:
            raise ValueError("sequence must be greater than zero")
        if not isinstance(self.lineage, UUID):
            raise TypeError("lineage must be a UUID")


@dataclass(frozen=True)
class RangeLock(Generic[LockOwnerT]):
    """One acquired half-open range lock."""

    handle: LockHandle[LockOwnerT]
    start: int
    end: int
    mode: LockMode
    insertion_order: int

    def __post_init__(self) -> None:
        Span(self.start, self.end)
        if not isinstance(self.mode, LockMode):
            raise TypeError("mode must be a LockMode")
        validate_coordinate(self.insertion_order, "insertion_order")
        if self.insertion_order < 0:
            raise ValueError("insertion_order cannot be negative")

    @property
    def owner(self) -> LockOwnerT:
        """Return the owner embedded in the lock handle."""
        return self.handle.owner

    @property
    def span(self) -> Span:
        """Return the locked range as a validated span."""
        return Span(self.start, self.end)


@dataclass(frozen=True)
class LockRequest(Generic[LockOwnerT]):
    """Validated lock request retained by conflict exceptions."""

    owner: LockOwnerT
    start: int
    end: int
    mode: LockMode

    def __post_init__(self) -> None:
        hash(self.owner)
        Span(self.start, self.end)
        if not isinstance(self.mode, LockMode):
            raise TypeError("mode must be a LockMode")


class LockConflictError(Exception, Generic[LockOwnerT]):
    """Raised with every exact incompatible overlapping lock."""

    def __init__(
        self,
        request: LockRequest[LockOwnerT],
        conflicts: tuple[RangeLock[LockOwnerT], ...],
    ) -> None:
        if not conflicts:
            raise ValueError("a lock conflict must identify at least one lock")
        self.request = request
        self.conflicts = conflicts
        detail = ", ".join(
            f"{lock.handle!r} [{lock.start}, {lock.end}) {lock.mode.value}"
            for lock in conflicts
        )
        super().__init__(
            f"{request.mode.value} lock [{request.start}, {request.end}) "
            f"for {request.owner!r} conflicts with {detail}"
        )


@dataclass(frozen=True)
class LockDiagnostics:
    """Payload-free lock table counters."""

    lock_count: int
    owner_count: int
    shared_count: int
    exclusive_count: int
    next_insertion_order: int


@dataclass(frozen=True)
class LockSnapshot(Generic[LockOwnerT]):
    """Immutable lock table state in deterministic acquisition order."""

    locks: tuple[RangeLock[LockOwnerT], ...]
    next_sequences: tuple[tuple[LockOwnerT, int], ...]
    next_insertion_order: int


class RangeLockTable(Generic[LockOwnerT]):
    """Thread-safe shared/exclusive interval locks.

    Locks belonging to the same owner are reentrant and never conflict with
    one another. They remain separate identities, so duplicate acquisitions
    require separate releases. Different owners conflict exactly when their
    half-open ranges overlap and at least one mode is exclusive.

    Handles are reconstructible value identities, not authorization
    capabilities. Upgrade and release require an explicit owner assertion,
    which callers must obtain from their own trusted authorization context.
    """

    def __init__(self) -> None:
        self._lineage = uuid4()
        self._locks: dict[LockHandle[LockOwnerT], RangeLock[LockOwnerT]] = {}
        self._next_sequences: dict[LockOwnerT, int] = {}
        self._next_insertion_order = 0
        self._lock = RLock()

    @staticmethod
    def _validate_owner(owner: LockOwnerT) -> None:
        try:
            hash(owner)
        except TypeError as exc:
            raise TypeError("owner must be hashable") from exc

    def _range_for_unlocked(
        self, handle: LockHandle[LockOwnerT]
    ) -> RangeLock[LockOwnerT]:
        if not isinstance(handle, LockHandle) or handle.lineage != self._lineage:
            raise KeyError(handle)
        try:
            return self._locks[handle]
        except KeyError:
            raise KeyError(handle) from None

    def _require_owner(
        self, handle: LockHandle[LockOwnerT], owner: LockOwnerT
    ) -> None:
        self._validate_owner(owner)
        if handle.owner != owner:
            raise PermissionError("lock belongs to another owner")

    @staticmethod
    def _coerce_mode(mode: LockMode | str) -> LockMode:
        try:
            return LockMode(mode)
        except (TypeError, ValueError) as exc:
            raise ValueError("mode must be 'shared' or 'exclusive'") from exc

    @staticmethod
    def _incompatible(requested: LockMode, existing: LockMode) -> bool:
        return requested is LockMode.EXCLUSIVE or existing is LockMode.EXCLUSIVE

    def _conflicts_unlocked(
        self,
        request: LockRequest[LockOwnerT],
        *,
        exclude: LockHandle[LockOwnerT] | None = None,
    ) -> tuple[RangeLock[LockOwnerT], ...]:
        return tuple(
            lock
            for lock in self._locks.values()
            if lock.handle != exclude
            and lock.owner != request.owner
            and lock.start < request.end
            and request.start < lock.end
            and self._incompatible(request.mode, lock.mode)
        )

    def conflicts(
        self,
        owner: LockOwnerT,
        start: int,
        end: int,
        mode: LockMode | str,
    ) -> tuple[RangeLock[LockOwnerT], ...]:
        """Return all incompatible locks in acquisition order."""
        self._validate_owner(owner)
        request = LockRequest(owner, start, end, self._coerce_mode(mode))
        with self._lock:
            return self._conflicts_unlocked(request)

    def acquire(
        self,
        owner: LockOwnerT,
        start: int,
        end: int,
        mode: LockMode | str,
    ) -> LockHandle[LockOwnerT]:
        """Acquire one lock or raise :class:`LockConflictError` atomically."""
        self._validate_owner(owner)
        request = LockRequest(owner, start, end, self._coerce_mode(mode))
        with self._lock:
            conflicts = self._conflicts_unlocked(request)
            if conflicts:
                raise LockConflictError(request, conflicts)
            sequence = self._next_sequences.get(owner, 1)
            handle = LockHandle(owner, sequence, self._lineage)
            acquired = RangeLock(
                handle,
                request.start,
                request.end,
                request.mode,
                self._next_insertion_order,
            )
            self._locks[handle] = acquired
            self._next_sequences[owner] = sequence + 1
            self._next_insertion_order += 1
            return handle

    def upgrade(
        self, handle: LockHandle[LockOwnerT], *, owner: LockOwnerT
    ) -> RangeLock[LockOwnerT]:
        """Upgrade a shared lock for an explicitly asserted lock owner."""
        with self._lock:
            current = self._range_for_unlocked(handle)
            self._require_owner(handle, owner)
            if current.mode is LockMode.EXCLUSIVE:
                return current
            request = LockRequest(
                current.owner, current.start, current.end, LockMode.EXCLUSIVE
            )
            conflicts = self._conflicts_unlocked(request, exclude=handle)
            if conflicts:
                raise LockConflictError(request, conflicts)
            upgraded = RangeLock(
                current.handle,
                current.start,
                current.end,
                LockMode.EXCLUSIVE,
                current.insertion_order,
            )
            self._locks[handle] = upgraded
            return upgraded

    def release(
        self, handle: LockHandle[LockOwnerT], *, owner: LockOwnerT
    ) -> RangeLock[LockOwnerT]:
        """Release one identity for an explicitly asserted lock owner."""
        with self._lock:
            current = self._range_for_unlocked(handle)
            self._require_owner(handle, owner)
            del self._locks[handle]
            return current

    def get(self, handle: LockHandle[LockOwnerT]) -> RangeLock[LockOwnerT]:
        """Return one immutable acquired lock."""
        with self._lock:
            return self._range_for_unlocked(handle)

    def diagnostics(self) -> LockDiagnostics:
        """Return aggregate active-lock counters."""
        with self._lock:
            shared_count = sum(
                lock.mode is LockMode.SHARED for lock in self._locks.values()
            )
            return LockDiagnostics(
                lock_count=len(self._locks),
                owner_count=len({lock.owner for lock in self._locks.values()}),
                shared_count=shared_count,
                exclusive_count=len(self._locks) - shared_count,
                next_insertion_order=self._next_insertion_order,
            )

    def snapshot(self) -> LockSnapshot[LockOwnerT]:
        """Return immutable active locks plus monotonic owner counters."""
        with self._lock:
            return LockSnapshot(
                locks=tuple(self._locks.values()),
                next_sequences=tuple(self._next_sequences.items()),
                next_insertion_order=self._next_insertion_order,
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._locks)
