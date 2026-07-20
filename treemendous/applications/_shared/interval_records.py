"""Identity-preserving interval records for private application engines.

Unlike a range set, this index never coalesces coincident records.  Each insert
receives an owner-scoped handle and a global insertion ordinal.  Query order is
that insertion order, including after an interval is updated.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock, RLock
from typing import Generic, TypeVar, cast

from treemendous.domain import Span, validate_coordinate

OwnerT = TypeVar("OwnerT", bound=Hashable)
PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True)
class RecordHandle(Generic[OwnerT]):
    """Stable owner-scoped identity for one interval record."""

    owner: OwnerT
    sequence: int

    def __post_init__(self) -> None:
        hash(self.owner)
        validate_coordinate(self.sequence, "sequence")
        if self.sequence < 1:
            raise ValueError("sequence must be greater than zero")


@dataclass(frozen=True)
class IntervalRecord(Generic[OwnerT, PayloadT]):
    """An immutable observation of an indexed record.

    ``payload`` is cloned at every boundary; mutating it cannot mutate index
    state when the supplied cloner honors its contract.
    """

    handle: RecordHandle[OwnerT]
    start: int
    end: int
    payload: PayloadT
    insertion_order: int

    def __post_init__(self) -> None:
        Span(self.start, self.end)
        validate_coordinate(self.insertion_order, "insertion_order")
        if self.insertion_order < 0:
            raise ValueError("insertion_order cannot be negative")

    @property
    def span(self) -> Span:
        """Return the record's validated half-open span."""
        return Span(self.start, self.end)

    @property
    def record_id(self) -> RecordHandle[OwnerT]:
        """Return the stable record identity."""
        return self.handle


@dataclass(frozen=True)
class IntervalRecordDiagnostics:
    """Cheap counters that do not expose payloads."""

    record_count: int
    owner_count: int
    next_insertion_order: int


@dataclass(frozen=True)
class IntervalRecordSnapshot(Generic[OwnerT, PayloadT]):
    """Detached, immutable observation of all records and owner counters."""

    records: tuple[IntervalRecord[OwnerT, PayloadT], ...]
    next_sequences: tuple[tuple[OwnerT, int], ...]
    next_insertion_order: int


_MISSING = object()


class IntervalRecordIndex(Generic[OwnerT, PayloadT]):
    """Thread-safe interval record index with explicit payload detachment.

    ``cloner`` is required rather than guessed.  It must not mutate its argument
    and must return a payload independently mutable from it (``copy.deepcopy``
    is a common choice).  A cloner exception leaves the index unchanged.
    """

    def __init__(self, cloner: Callable[[PayloadT], PayloadT]) -> None:
        if not callable(cloner):
            raise TypeError("cloner must be callable")
        self._cloner = cloner
        self._records: dict[RecordHandle[OwnerT], IntervalRecord[OwnerT, PayloadT]] = {}
        self._next_sequences: dict[OwnerT, int] = {}
        self._next_insertion_order = 0
        self._lock = RLock()
        self._payload_activity_lock = Lock()
        self._payload_activity = 0

    @staticmethod
    def _validate_owner(owner: OwnerT) -> None:
        try:
            hash(owner)
        except TypeError as exc:
            raise TypeError("owner must be hashable") from exc

    @contextmanager
    def _payload_copying(self) -> Iterator[None]:
        with self._payload_activity_lock:
            self._payload_activity += 1
        try:
            yield
        finally:
            with self._payload_activity_lock:
                self._payload_activity -= 1

    def _payload_is_active(self) -> bool:
        with self._payload_activity_lock:
            return self._payload_activity > 0

    @contextmanager
    def _mutation(self) -> Iterator[None]:
        # Do not deadlock when a cloner waits for another thread that attempts
        # to mutate this index.  Mutations from any cloner are rejected.
        while True:
            if self._payload_is_active():
                raise RuntimeError(
                    "IntervalRecordIndex mutation is not allowed during payload cloning"
                )
            if self._lock.acquire(timeout=0.01):
                break
        try:
            if self._payload_is_active():
                raise RuntimeError(
                    "IntervalRecordIndex mutation is not allowed during payload cloning"
                )
            yield
        finally:
            self._lock.release()

    def _detach(
        self, record: IntervalRecord[OwnerT, PayloadT]
    ) -> IntervalRecord[OwnerT, PayloadT]:
        return IntervalRecord(
            record.handle,
            record.start,
            record.end,
            self._cloner(record.payload),
            record.insertion_order,
        )

    def insert(
        self, owner: OwnerT, start: int, end: int, payload: PayloadT
    ) -> RecordHandle[OwnerT]:
        """Insert one record and return a new owner-scoped handle."""
        self._validate_owner(owner)
        span = Span(start, end)
        with self._mutation():
            # Clone before changing either monotonic counter.
            with self._payload_copying():
                stored_payload = self._cloner(payload)
            sequence = self._next_sequences.get(owner, 1)
            handle = RecordHandle(owner, sequence)
            record = IntervalRecord(
                handle,
                span.start,
                span.end,
                stored_payload,
                self._next_insertion_order,
            )
            self._records[handle] = record
            self._next_sequences[owner] = sequence + 1
            self._next_insertion_order += 1
            return handle

    def update(
        self,
        handle: RecordHandle[OwnerT],
        *,
        start: int | None = None,
        end: int | None = None,
        payload: PayloadT | object = _MISSING,
    ) -> IntervalRecord[OwnerT, PayloadT]:
        """Atomically replace selected fields without changing identity/order."""
        with self._mutation():
            current = self._records.get(handle)
            if current is None:
                raise KeyError(handle)
            if start is None and end is None and payload is _MISSING:
                raise ValueError("update requires a span or payload replacement")
            span = Span(
                current.start if start is None else start,
                current.end if end is None else end,
            )
            with self._payload_copying():
                if payload is _MISSING:
                    stored_payload = self._cloner(current.payload)
                else:
                    stored_payload = self._cloner(cast(PayloadT, payload))
                candidate = IntervalRecord(
                    handle,
                    span.start,
                    span.end,
                    stored_payload,
                    current.insertion_order,
                )
                # Prepare the returned detached value before committing.
                result = self._detach(candidate)
            self._records[handle] = candidate
            return result

    def remove(self, handle: RecordHandle[OwnerT]) -> IntervalRecord[OwnerT, PayloadT]:
        """Remove exactly one identity and return a detached prior value."""
        with self._mutation():
            current = self._records.get(handle)
            if current is None:
                raise KeyError(handle)
            with self._payload_copying():
                result = self._detach(current)
            del self._records[handle]
            return result

    def get(self, handle: RecordHandle[OwnerT]) -> IntervalRecord[OwnerT, PayloadT]:
        """Return a detached record for ``handle``."""
        with self._lock:
            current = self._records.get(handle)
            if current is None:
                raise KeyError(handle)
            with self._payload_copying():
                return self._detach(current)

    def overlaps(
        self, start: int, end: int
    ) -> tuple[IntervalRecord[OwnerT, PayloadT], ...]:
        """Return records intersecting ``[start, end)`` in insertion order."""
        span = Span(start, end)
        with self._lock:
            with self._payload_copying():
                return tuple(
                    self._detach(record)
                    for record in self._records.values()
                    if record.start < span.end and span.start < record.end
                )

    def containing(self, point: int) -> tuple[IntervalRecord[OwnerT, PayloadT], ...]:
        """Return records containing ``point`` under half-open semantics."""
        validate_coordinate(point, "point")
        with self._lock:
            with self._payload_copying():
                return tuple(
                    self._detach(record)
                    for record in self._records.values()
                    if record.start <= point < record.end
                )

    def at(self, point: int) -> tuple[IntervalRecord[OwnerT, PayloadT], ...]:
        """Alias for :meth:`containing` used by point-querying applications."""
        return self.containing(point)

    def diagnostics(self) -> IntervalRecordDiagnostics:
        """Return payload-free counters."""
        with self._lock:
            return IntervalRecordDiagnostics(
                record_count=len(self._records),
                owner_count=len(self._next_sequences),
                next_insertion_order=self._next_insertion_order,
            )

    def snapshot(self) -> IntervalRecordSnapshot[OwnerT, PayloadT]:
        """Return a detached snapshot ordered by original insertion."""
        with self._lock:
            with self._payload_copying():
                records = tuple(
                    self._detach(record) for record in self._records.values()
                )
            return IntervalRecordSnapshot(
                records=records,
                next_sequences=tuple(self._next_sequences.items()),
                next_insertion_order=self._next_insertion_order,
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
