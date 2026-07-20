"""Deterministic contiguous allocation over an explicit integer domain.

This private kernel deliberately owns its pure-Python ``RangeSet`` backend.  It
keeps policy, identity, and checkpoint semantics here instead of leaking them
into application engines.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from uuid import uuid4

from treemendous.backends.adapters import BackendAdapter
from treemendous.basic.boundary import IntervalManager
from treemendous.domain import (
    DomainInput,
    IntervalResult,
    ManagedDomain,
    Span,
    validate_coordinate,
    validate_length,
)
from treemendous.rangeset import RangeSet


class AllocationError(Exception):
    """Base error for allocator state transitions."""


class AllocationUnavailableError(AllocationError):
    """Raised when the requested contiguous extent is unavailable."""


class AllocationConflictError(AllocationError):
    """Raised when an idempotency key is reused for a different request."""


class ForeignAllocationError(AllocationError):
    """Raised when a handle belongs to another allocator or owner."""


class StaleAllocationError(AllocationError):
    """Raised when an allocation handle is no longer live."""


class FitPolicy(str, Enum):
    """Deterministic free-chunk selection policy."""

    FIRST = "first"
    BEST = "best"
    WORST = "worst"


@dataclass(frozen=True)
class AllocationHandle:
    """Opaque evidence of one owner-scoped live allocation."""

    allocator_id: str
    allocation_id: int
    owner: Hashable
    span: Span

    @property
    def start(self) -> int:
        return self.span.start

    @property
    def end(self) -> int:
        return self.span.end

    @property
    def size(self) -> int:
        return self.span.length


@dataclass(frozen=True)
class FragmentationDiagnostics:
    """Occupancy and external-fragmentation measurements."""

    total_space: int
    allocated_space: int
    reserved_space: int
    total_free: int
    free_chunks: int
    largest_free_chunk: int
    average_free_chunk: float
    fragmentation: float


@dataclass(frozen=True)
class AllocatorSnapshot:
    """Immutable observable allocator state."""

    domain: ManagedDomain
    free_ranges: tuple[Span, ...]
    reserved_ranges: tuple[Span, ...]
    allocations: tuple[AllocationHandle, ...]
    diagnostics: FragmentationDiagnostics


@dataclass(frozen=True)
class _AllocationRecord:
    handle: AllocationHandle
    size: int
    alignment: int
    policy: FitPolicy | None
    exact_start: int | None
    idempotency_key: Hashable | None


@dataclass(frozen=True)
class AllocatorCheckpoint:
    """Complete immutable state accepted by :meth:`ContiguousAllocator.restore`."""

    allocator_id: str
    domain: ManagedDomain
    free_ranges: tuple[Span, ...]
    reserved_ranges: tuple[Span, ...]
    records: tuple[_AllocationRecord, ...]
    idempotency: tuple[tuple[Hashable, Hashable, int], ...]
    next_allocation_id: int


def _new_range_set(domain: ManagedDomain) -> RangeSet:
    return RangeSet(
        BackendAdapter(IntervalManager()),
        domain=domain,
        initially_available=True,
    )


def _coerce_span(value: Span | tuple[int, int]) -> Span:
    if isinstance(value, Span):
        return value
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("reserved range must be a Span or (start, end) tuple")
    return Span(value[0], value[1])


def _normalize_spans(spans: Iterable[Span]) -> tuple[Span, ...]:
    merged: list[Span] = []
    for span in sorted(spans):
        if merged and span.start <= merged[-1].end:
            merged[-1] = Span(merged[-1].start, max(merged[-1].end, span.end))
        else:
            merged.append(span)
    return tuple(merged)


def _hashable(value: object, name: str) -> Hashable:
    try:
        hash(value)
    except TypeError as error:
        raise TypeError(f"{name} must be hashable") from error
    return value


def _align_up(value: int, alignment: int) -> int:
    # Floor division makes this correct on both sides of zero.
    return -((-value) // alignment) * alignment


class ContiguousAllocator:
    """Thread-safe allocator for half-open spans in an explicit managed domain.

    Alignment is relative to integer zero.  Best and worst fit rank complete
    free chunks by length, breaking ties by the aligned candidate coordinate.
    Only the requested extent is removed; alignment padding remains free.
    """

    def __init__(
        self,
        domain: DomainInput,
        *,
        reserved: Iterable[Span | tuple[int, int]] = (),
    ) -> None:
        self._domain = domain if isinstance(domain, ManagedDomain) else ManagedDomain(domain)
        staged_free = _new_range_set(self._domain)
        reserved_ranges = _normalize_spans(_coerce_span(item) for item in reserved)
        for span in reserved_ranges:
            if not self._domain.contains(span):
                raise ValueError("reserved range must be contained in the managed domain")
            staged_free.discard(span, require_covered=True)

        self._lock = RLock()
        self._allocator_id = uuid4().hex
        self._free = staged_free
        self._reserved = reserved_ranges
        self._records: dict[int, _AllocationRecord] = {}
        self._idempotency: dict[tuple[Hashable, Hashable], int] = {}
        self._next_allocation_id = 1

    @property
    def domain(self) -> ManagedDomain:
        return self._domain

    def allocate(
        self,
        size: int,
        *,
        owner: Hashable,
        alignment: int = 1,
        policy: FitPolicy | str = FitPolicy.FIRST,
        idempotency_key: Hashable | None = None,
    ) -> AllocationHandle:
        """Allocate a policy-selected aligned extent of exactly ``size``."""
        validate_length(size)
        alignment = self._validate_alignment(alignment)
        owner = _hashable(owner, "owner")
        selected_policy = self._coerce_policy(policy)
        key = self._idempotency_key(owner, idempotency_key)

        with self._lock:
            prior = self._prior_record(key)
            if prior is not None:
                if (
                    prior.size,
                    prior.alignment,
                    prior.policy,
                    prior.exact_start,
                ) != (size, alignment, selected_policy, None):
                    raise AllocationConflictError(
                        "idempotency key was already used for a different request"
                    )
                return prior.handle

            candidate = self._select_candidate(size, alignment, selected_policy)
            if candidate is None:
                raise AllocationUnavailableError(
                    "no free chunk satisfies size and alignment"
                )
            return self._commit(
                Span(candidate, candidate + size),
                owner=owner,
                alignment=alignment,
                policy=selected_policy,
                exact_start=None,
                idempotency_key=idempotency_key,
                idempotency_map_key=key,
            )

    def reserve(
        self,
        start: int,
        size: int,
        *,
        owner: Hashable,
        alignment: int = 1,
        idempotency_key: Hashable | None = None,
    ) -> AllocationHandle:
        """Reserve exactly ``[start, start + size)`` or fail without mutation."""
        validate_coordinate(start, "start")
        validate_length(size)
        alignment = self._validate_alignment(alignment)
        if start % alignment:
            raise ValueError("start must satisfy alignment")
        owner = _hashable(owner, "owner")
        key = self._idempotency_key(owner, idempotency_key)
        requested = Span(start, start + size)
        if not self._domain.contains(requested):
            raise ValueError("requested span must be contained in the managed domain")

        with self._lock:
            prior = self._prior_record(key)
            if prior is not None:
                if (
                    prior.size,
                    prior.alignment,
                    prior.policy,
                    prior.exact_start,
                ) != (size, alignment, None, start):
                    raise AllocationConflictError(
                        "idempotency key was already used for a different request"
                    )
                return prior.handle
            return self._commit(
                requested,
                owner=owner,
                alignment=alignment,
                policy=None,
                exact_start=start,
                idempotency_key=idempotency_key,
                idempotency_map_key=key,
            )

    def reserve_hole(self, span: Span | tuple[int, int]) -> None:
        """Permanently remove an unowned hole from the allocatable domain."""
        requested = _coerce_span(span)
        if not self._domain.contains(requested):
            raise ValueError("reserved range must be contained in the managed domain")
        with self._lock:
            staged_reserved = _normalize_spans((*self._reserved, requested))
            mutation = self._free.discard(requested, require_covered=True)
            if not mutation.fully_covered:
                raise AllocationUnavailableError("reserved range is not entirely free")
            self._reserved = staged_reserved

    def free(
        self,
        handle: AllocationHandle,
        *,
        owner: Hashable | None = None,
    ) -> None:
        """Free a live handle, rejecting stale, forged, foreign, or wrong-owner use."""
        if not isinstance(handle, AllocationHandle):
            raise TypeError("handle must be an AllocationHandle")
        if owner is not None:
            owner = _hashable(owner, "owner")
        with self._lock:
            if handle.allocator_id != self._allocator_id:
                raise ForeignAllocationError("allocation belongs to another allocator")
            if owner is not None and owner != handle.owner:
                raise ForeignAllocationError("allocation belongs to another owner")
            record = self._records.get(handle.allocation_id)
            if record is None or record.handle != handle:
                raise StaleAllocationError("allocation handle is stale or invalid")

            staged_records = dict(self._records)
            del staged_records[handle.allocation_id]
            staged_idempotency = dict(self._idempotency)
            if record.idempotency_key is not None:
                del staged_idempotency[(handle.owner, record.idempotency_key)]

            # Build metadata first; RangeSet.add is then the final fallible
            # transition before reference assignments commit the new state.
            self._free.add(handle.span)
            self._records = staged_records
            self._idempotency = staged_idempotency

    def diagnostics(self) -> FragmentationDiagnostics:
        """Return deterministic external-fragmentation diagnostics."""
        with self._lock:
            free_ranges = self._free.intervals()
            total_free = sum(item.end - item.start for item in free_ranges)
            largest = max((item.end - item.start for item in free_ranges), default=0)
            chunks = len(free_ranges)
            allocated = sum(record.handle.size for record in self._records.values())
            reserved = sum(span.length for span in self._reserved)
            if allocated + reserved + total_free != self._domain.measure:
                raise RuntimeError("allocator accounting invariant was violated")
            return FragmentationDiagnostics(
                total_space=self._domain.measure,
                allocated_space=allocated,
                reserved_space=reserved,
                total_free=total_free,
                free_chunks=chunks,
                largest_free_chunk=largest,
                average_free_chunk=total_free / chunks if chunks else 0.0,
                fragmentation=(1.0 - largest / total_free) if total_free else 0.0,
            )

    def snapshot(self) -> AllocatorSnapshot:
        """Return an immutable, coordinate-ordered view of current state."""
        with self._lock:
            return AllocatorSnapshot(
                domain=self._domain,
                free_ranges=self._free_spans(),
                reserved_ranges=self._reserved,
                allocations=tuple(
                    record.handle
                    for record in sorted(
                        self._records.values(), key=lambda item: item.handle.span
                    )
                ),
                diagnostics=self.diagnostics(),
            )

    def checkpoint(self) -> AllocatorCheckpoint:
        """Capture complete state suitable for later restoration on this allocator."""
        with self._lock:
            return AllocatorCheckpoint(
                allocator_id=self._allocator_id,
                domain=self._domain,
                free_ranges=self._free_spans(),
                reserved_ranges=self._reserved,
                records=tuple(
                    self._records[key] for key in sorted(self._records)
                ),
                idempotency=tuple(
                    (owner, key, allocation_id)
                    for (owner, key), allocation_id in self._idempotency.items()
                ),
                next_allocation_id=self._next_allocation_id,
            )

    def restore(self, checkpoint: AllocatorCheckpoint) -> None:
        """Atomically restore a checkpoint created by this allocator."""
        if not isinstance(checkpoint, AllocatorCheckpoint):
            raise TypeError("checkpoint must be an AllocatorCheckpoint")
        with self._lock:
            if checkpoint.allocator_id != self._allocator_id:
                raise ForeignAllocationError("checkpoint belongs to another allocator")
            if checkpoint.domain != self._domain:
                raise ValueError("checkpoint domain does not match allocator domain")
            staged = self._validate_checkpoint(checkpoint)
            self._free = staged
            self._reserved = checkpoint.reserved_ranges
            self._records = {
                record.handle.allocation_id: record for record in checkpoint.records
            }
            self._idempotency = {
                (owner, key): allocation_id
                for owner, key, allocation_id in checkpoint.idempotency
            }
            self._next_allocation_id = checkpoint.next_allocation_id

    @staticmethod
    def _validate_alignment(alignment: int) -> int:
        validate_coordinate(alignment, "alignment")
        if alignment <= 0:
            raise ValueError("alignment must be greater than zero")
        return alignment

    @staticmethod
    def _coerce_policy(policy: FitPolicy | str) -> FitPolicy:
        try:
            return FitPolicy(policy)
        except (TypeError, ValueError) as error:
            raise ValueError("policy must be first, best, or worst") from error

    @staticmethod
    def _idempotency_key(
        owner: Hashable, key: Hashable | None
    ) -> tuple[Hashable, Hashable] | None:
        if key is None:
            return None
        return owner, _hashable(key, "idempotency_key")

    def _prior_record(
        self, key: tuple[Hashable, Hashable] | None
    ) -> _AllocationRecord | None:
        if key is None:
            return None
        allocation_id = self._idempotency.get(key)
        if allocation_id is None:
            return None
        record = self._records.get(allocation_id)
        if record is None:
            raise RuntimeError("idempotency index invariant was violated")
        return record

    def _select_candidate(
        self, size: int, alignment: int, policy: FitPolicy
    ) -> int | None:
        candidates: list[tuple[IntervalResult, int]] = []
        for chunk in self._free.intervals():
            candidate = _align_up(chunk.start, alignment)
            if candidate + size <= chunk.end:
                candidates.append((chunk, candidate))
        if not candidates:
            return None
        if policy is FitPolicy.FIRST:
            return min(candidate for _, candidate in candidates)
        if policy is FitPolicy.BEST:
            return min(
                candidates,
                key=lambda item: (item[0].end - item[0].start, item[1]),
            )[1]
        return min(
            candidates,
            key=lambda item: (-(item[0].end - item[0].start), item[1]),
        )[1]

    def _commit(
        self,
        span: Span,
        *,
        owner: Hashable,
        alignment: int,
        policy: FitPolicy | None,
        exact_start: int | None,
        idempotency_key: Hashable | None,
        idempotency_map_key: tuple[Hashable, Hashable] | None,
    ) -> AllocationHandle:
        allocation_id = self._next_allocation_id
        handle = AllocationHandle(
            allocator_id=self._allocator_id,
            allocation_id=allocation_id,
            owner=owner,
            span=span,
        )
        record = _AllocationRecord(
            handle=handle,
            size=span.length,
            alignment=alignment,
            policy=policy,
            exact_start=exact_start,
            idempotency_key=idempotency_key,
        )
        staged_records = dict(self._records)
        staged_records[allocation_id] = record
        staged_idempotency = dict(self._idempotency)
        if idempotency_map_key is not None:
            staged_idempotency[idempotency_map_key] = allocation_id

        mutation = self._free.discard(span, require_covered=True)
        if not mutation.fully_covered:
            raise AllocationUnavailableError("requested span is not entirely free")
        self._records = staged_records
        self._idempotency = staged_idempotency
        self._next_allocation_id = allocation_id + 1
        return handle

    def _free_spans(self) -> tuple[Span, ...]:
        return tuple(item.span for item in self._free.intervals())

    def _validate_checkpoint(self, checkpoint: AllocatorCheckpoint) -> RangeSet:
        validate_coordinate(
            checkpoint.next_allocation_id, "checkpoint allocation counter"
        )
        if checkpoint.next_allocation_id <= 0:
            raise ValueError("checkpoint allocation counter must be positive")
        if checkpoint.reserved_ranges != _normalize_spans(checkpoint.reserved_ranges):
            raise ValueError("checkpoint reserved ranges are not normalized")

        staged = _new_range_set(self._domain)
        for span in checkpoint.reserved_ranges:
            if not self._domain.contains(span):
                raise ValueError("checkpoint reserved range is outside domain")
            staged.discard(span, require_covered=True)

        records: dict[int, _AllocationRecord] = {}
        for record in checkpoint.records:
            handle = record.handle
            if handle.allocator_id != self._allocator_id:
                raise ForeignAllocationError("checkpoint contains a foreign handle")
            validate_coordinate(handle.allocation_id, "checkpoint allocation ID")
            if handle.allocation_id <= 0:
                raise ValueError("checkpoint allocation ID must be positive")
            if handle.allocation_id in records:
                raise ValueError("checkpoint contains duplicate allocation IDs")
            _hashable(handle.owner, "checkpoint owner")
            alignment = self._validate_alignment(record.alignment)
            if record.size != handle.size:
                raise ValueError("checkpoint contains invalid allocation metadata")
            if handle.start % alignment:
                raise ValueError("checkpoint allocation violates alignment")
            if record.policy is None:
                if record.exact_start != handle.start:
                    raise ValueError("checkpoint exact reservation metadata is invalid")
            elif not isinstance(record.policy, FitPolicy) or record.exact_start is not None:
                raise ValueError("checkpoint policy allocation metadata is invalid")
            if record.idempotency_key is not None:
                _hashable(record.idempotency_key, "checkpoint idempotency key")
            if not self._domain.contains(handle.span):
                raise ValueError("checkpoint allocation is outside domain")
            if not staged.discard(handle.span, require_covered=True).fully_covered:
                raise ValueError("checkpoint allocations overlap occupied space")
            records[handle.allocation_id] = record

        if records and checkpoint.next_allocation_id <= max(records):
            raise ValueError("checkpoint allocation counter is stale")
        rebuilt_idempotency: dict[tuple[Hashable, Hashable], int] = {}
        for owner, key, allocation_id in checkpoint.idempotency:
            map_key = (_hashable(owner, "checkpoint owner"), _hashable(key, "checkpoint key"))
            if map_key in rebuilt_idempotency:
                raise ValueError("checkpoint contains duplicate idempotency keys")
            indexed_record = records.get(allocation_id)
            if (
                indexed_record is None
                or indexed_record.handle.owner != owner
                or indexed_record.idempotency_key != key
            ):
                raise ValueError("checkpoint idempotency index is inconsistent")
            rebuilt_idempotency[map_key] = allocation_id
        expected_ids = {
            record.handle.allocation_id
            for record in records.values()
            if record.idempotency_key is not None
        }
        if expected_ids != set(rebuilt_idempotency.values()):
            raise ValueError("checkpoint is missing idempotency entries")
        if tuple(item.span for item in staged.intervals()) != checkpoint.free_ranges:
            raise ValueError("checkpoint free geometry is inconsistent")
        return staged
