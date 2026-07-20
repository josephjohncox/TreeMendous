"""Identity-preserving interval records for private application engines.

Unlike a range set, this index never coalesces coincident records.  Each insert
receives an index-scoped handle and a global insertion ordinal.  Query order is
that insertion order, including after an interval is updated.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Generic, TypeVar, cast
from uuid import UUID, uuid4

from treemendous.domain import Span, validate_coordinate

OwnerT = TypeVar("OwnerT", bound=Hashable)
PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True)
class RecordHandle(Generic[OwnerT]):
    """Index-scoped value identity, not authorization for a record."""

    owner: OwnerT
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

    ``cloner`` is required rather than guessed. It must not mutate its argument
    and must return a payload independently mutable from it (``copy.deepcopy``
    is a common choice). Cloning never runs under the state lock. A cloner
    exception leaves the pending operation uncommitted; any reentrant operation
    initiated by the cloner is independent and linearizes before it.

    Handles are reconstructible value identities, not authorization
    capabilities. Mutating methods require an explicit owner assertion, which
    callers must obtain from their own trusted authorization context.
    """

    def __init__(self, cloner: Callable[[PayloadT], PayloadT]) -> None:
        if not callable(cloner):
            raise TypeError("cloner must be callable")
        self._cloner = cloner
        self._lineage = uuid4()
        self._records: dict[RecordHandle[OwnerT], IntervalRecord[OwnerT, PayloadT]] = {}
        self._next_sequences: dict[OwnerT, int] = {}
        self._next_insertion_order = 0
        self._version = 0
        self._lock = RLock()

    @staticmethod
    def _validate_owner(owner: OwnerT) -> None:
        try:
            hash(owner)
        except TypeError as exc:
            raise TypeError("owner must be hashable") from exc

    def _record_for_unlocked(
        self, handle: RecordHandle[OwnerT]
    ) -> IntervalRecord[OwnerT, PayloadT]:
        if not isinstance(handle, RecordHandle) or handle.lineage != self._lineage:
            raise KeyError(handle)
        try:
            return self._records[handle]
        except KeyError:
            raise KeyError(handle) from None

    def _require_owner(self, handle: RecordHandle[OwnerT], owner: OwnerT) -> None:
        self._validate_owner(owner)
        if handle.owner != owner:
            raise PermissionError("record belongs to another owner")

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
        """Insert one record and return a new index-scoped handle."""
        self._validate_owner(owner)
        span = Span(start, end)
        # Arbitrary cloner code runs before the state commit. Reentrant effects
        # therefore linearize before this insertion.
        stored_payload = self._cloner(payload)
        with self._lock:
            sequence = self._next_sequences.get(owner, 1)
            handle = RecordHandle(owner, sequence, self._lineage)
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
            self._version += 1
            return handle

    def update(
        self,
        handle: RecordHandle[OwnerT],
        *,
        owner: OwnerT,
        start: int | None = None,
        end: int | None = None,
        payload: PayloadT | object = _MISSING,
    ) -> IntervalRecord[OwnerT, PayloadT]:
        """Replace selected fields for an explicitly asserted record owner.

        Payload preparation happens outside the state lock. If this identity is
        concurrently changed, the operation retries from that newer record;
        unrelated mutations do not invalidate its prepared candidate.
        """
        if start is None and end is None and payload is _MISSING:
            raise ValueError("update requires a span or payload replacement")
        while True:
            with self._lock:
                current = self._record_for_unlocked(handle)
                self._require_owner(handle, owner)
                version = self._version
            span = Span(
                current.start if start is None else start,
                current.end if end is None else end,
            )
            source_payload = (
                current.payload if payload is _MISSING else cast(PayloadT, payload)
            )
            stored_payload = self._cloner(source_payload)
            candidate = IntervalRecord(
                handle,
                span.start,
                span.end,
                stored_payload,
                current.insertion_order,
            )
            # Prepare the detached return before commit for failure atomicity.
            result = self._detach(candidate)
            with self._lock:
                latest = self._record_for_unlocked(handle)
                if self._version != version and latest is not current:
                    continue
                self._records[handle] = candidate
                self._version += 1
                return result

    def replace_batch(
        self,
        *,
        updates: Sequence[tuple[RecordHandle[OwnerT], OwnerT, int, int, PayloadT]] = (),
        removals: Sequence[tuple[RecordHandle[OwnerT], OwnerT]] = (),
    ) -> tuple[IntervalRecord[OwnerT, PayloadT], ...]:
        """Atomically replace and remove a validated batch of records.

        Every span, owner assertion, payload clone, and detached return value is
        prepared before the single commit. A preparation failure leaves every
        record unchanged. If a target changes concurrently during preparation,
        the whole batch is rejected rather than partially applied.
        """
        update_items = tuple(updates)
        removal_items = tuple(removals)
        handles = [item[0] for item in update_items]
        handles.extend(item[0] for item in removal_items)
        if len(set(handles)) != len(handles):
            raise ValueError("a record can appear only once in a batch")
        if not handles:
            return ()

        with self._lock:
            captured: dict[RecordHandle[OwnerT], IntervalRecord[OwnerT, PayloadT]] = {}
            for handle, owner, *_rest in update_items:
                current = self._record_for_unlocked(handle)
                self._require_owner(handle, owner)
                captured[handle] = current
            for handle, owner in removal_items:
                current = self._record_for_unlocked(handle)
                self._require_owner(handle, owner)
                captured[handle] = current

        candidates: list[IntervalRecord[OwnerT, PayloadT]] = []
        results: list[IntervalRecord[OwnerT, PayloadT]] = []
        for handle, _owner, start, end, payload in update_items:
            current = captured[handle]
            span = Span(start, end)
            candidate = IntervalRecord(
                handle,
                span.start,
                span.end,
                self._cloner(payload),
                current.insertion_order,
            )
            candidates.append(candidate)
            results.append(self._detach(candidate))
        for handle, _owner in removal_items:
            results.append(self._detach(captured[handle]))

        with self._lock:
            if any(
                self._records.get(handle) is not current
                for handle, current in captured.items()
            ):
                raise RuntimeError("a batch target changed during preparation")
            for candidate in candidates:
                self._records[candidate.handle] = candidate
            for handle, _owner in removal_items:
                del self._records[handle]
            self._version += 1
        return tuple(results)

    def remove(
        self, handle: RecordHandle[OwnerT], *, owner: OwnerT
    ) -> IntervalRecord[OwnerT, PayloadT]:
        """Remove one identity for an explicitly asserted record owner."""
        while True:
            with self._lock:
                current = self._record_for_unlocked(handle)
                self._require_owner(handle, owner)
                version = self._version
            result = self._detach(current)
            with self._lock:
                latest = self._record_for_unlocked(handle)
                if self._version != version and latest is not current:
                    continue
                del self._records[handle]
                self._version += 1
                return result

    def get(self, handle: RecordHandle[OwnerT]) -> IntervalRecord[OwnerT, PayloadT]:
        """Return a detached record for ``handle``."""
        with self._lock:
            current = self._record_for_unlocked(handle)
        return self._detach(current)

    def overlaps(
        self, start: int, end: int
    ) -> tuple[IntervalRecord[OwnerT, PayloadT], ...]:
        """Return records intersecting ``[start, end)`` in insertion order."""
        span = Span(start, end)
        with self._lock:
            captured = tuple(
                record
                for record in self._records.values()
                if record.start < span.end and span.start < record.end
            )
        return tuple(self._detach(record) for record in captured)

    def containing(self, point: int) -> tuple[IntervalRecord[OwnerT, PayloadT], ...]:
        """Return records containing ``point`` under half-open semantics."""
        validate_coordinate(point, "point")
        with self._lock:
            captured = tuple(
                record
                for record in self._records.values()
                if record.start <= point < record.end
            )
        return tuple(self._detach(record) for record in captured)

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
            captured = tuple(self._records.values())
            next_sequences = tuple(self._next_sequences.items())
            next_insertion_order = self._next_insertion_order
        records = tuple(self._detach(record) for record in captured)
        return IntervalRecordSnapshot(
            records=records,
            next_sequences=next_sequences,
            next_insertion_order=next_insertion_order,
        )

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
