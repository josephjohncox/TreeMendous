"""Deterministic in-memory resource reservation state machine.

The ledger provides atomicity only inside one Python process.  A multi-resource
reservation is validated and staged under one lock, then exposed with a single
state update; failed validation exposes none of it.  This is not a distributed
transaction, consensus protocol, or durable database.

Unlike a binary ``RangeSet``, resource capacity can be greater than one and can
have several dimensions.  This module therefore uses a sweep-line cumulative
capacity check rather than pretending binary interval availability is a
cumulative scheduler.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from threading import RLock
from typing import TypeAlias

from treemendous.applications._shared.capacity import CapacityVector
from treemendous.domain import Span, validate_coordinate, validate_length

CapacityInput: TypeAlias = CapacityVector | Mapping[str, int]
RequirementInput: TypeAlias = Mapping[str, CapacityInput]


def _non_empty_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _non_negative_integer(value: int, name: str) -> int:
    validate_coordinate(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _owner_sequence(reservation_id: str, owner: str) -> int:
    prefix = f"{owner}:"
    if not reservation_id.startswith(prefix):
        raise ValueError("reservation ID does not match its owner")
    suffix = reservation_id[len(prefix) :]
    try:
        sequence = int(suffix)
    except ValueError:
        raise ValueError("reservation ID has an invalid owner sequence") from None
    if sequence <= 0 or str(sequence) != suffix:
        raise ValueError("reservation ID has an invalid owner sequence")
    return sequence


@dataclass(frozen=True)
class ResourceRequirement:
    """Capacity required from one named resource."""

    resource: str
    capacity: CapacityVector

    def __post_init__(self) -> None:
        _non_empty_text(self.resource, "resource")
        if not isinstance(self.capacity, CapacityVector):
            raise TypeError("capacity must be a CapacityVector")


@dataclass(frozen=True)
class ResourceCapacity:
    """Capacity definition for one named resource."""

    resource: str
    capacity: CapacityVector

    def __post_init__(self) -> None:
        _non_empty_text(self.resource, "resource")
        if not isinstance(self.capacity, CapacityVector):
            raise TypeError("capacity must be a CapacityVector")


class ReservationStatus(str, Enum):
    """Lifecycle states retained by the in-memory ledger."""

    ACTIVE = "active"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Reservation:
    """Immutable reservation identity and its user-visible service window."""

    reservation_id: str
    owner: str
    start: int
    end: int
    requirements: tuple[ResourceRequirement, ...]
    buffer_before: int = 0
    buffer_after: int = 0
    request_id: str | None = None
    status: ReservationStatus = ReservationStatus.ACTIVE

    def __post_init__(self) -> None:
        _non_empty_text(self.reservation_id, "reservation_id")
        _non_empty_text(self.owner, "owner")
        Span(self.start, self.end)
        _non_negative_integer(self.buffer_before, "buffer_before")
        _non_negative_integer(self.buffer_after, "buffer_after")
        if self.request_id is not None:
            _non_empty_text(self.request_id, "request_id")
        if not self.requirements:
            raise ValueError("a reservation must require at least one resource")
        resources = tuple(item.resource for item in self.requirements)
        if resources != tuple(sorted(resources)) or len(resources) != len(set(resources)):
            raise ValueError("requirements must contain unique resources in sorted order")
        if not isinstance(self.status, ReservationStatus):
            raise TypeError("status must be a ReservationStatus")

    @property
    def id(self) -> str:
        """Short alias for ``reservation_id``."""
        return self.reservation_id

    @property
    def span(self) -> Span:
        """Return the requested service interval."""
        return Span(self.start, self.end)

    @property
    def occupied_span(self) -> Span:
        """Return service plus turnaround/buffer occupancy."""
        return Span(self.start - self.buffer_before, self.end + self.buffer_after)

    @property
    def active(self) -> bool:
        return self.status is ReservationStatus.ACTIVE


@dataclass(frozen=True)
class ReservationConflict:
    """One maximal time segment whose cumulative demand exceeds capacity."""

    resource: str
    start: int
    end: int
    capacity: CapacityVector
    used: CapacityVector
    requested: CapacityVector
    reservation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _non_empty_text(self.resource, "resource")
        Span(self.start, self.end)
        self.capacity._require_same_dimensions(self.used)
        self.capacity._require_same_dimensions(self.requested)
        if tuple(sorted(self.reservation_ids)) != self.reservation_ids:
            raise ValueError("conflicting reservation IDs must be sorted")


class ReservationConflictError(ValueError):
    """Raised with deterministic diagnostics when capacity is unavailable."""

    def __init__(self, conflicts: tuple[ReservationConflict, ...]) -> None:
        if not conflicts:
            raise ValueError("ReservationConflictError requires conflict details")
        self.conflicts = conflicts
        first = conflicts[0]
        super().__init__(
            f"resource {first.resource!r} lacks capacity during "
            f"[{first.start}, {first.end})"
        )


@dataclass(frozen=True)
class ReservationSnapshot:
    """Immutable deterministic observable ledger state."""

    resources: tuple[ResourceCapacity, ...]
    reservations: tuple[Reservation, ...]


@dataclass(frozen=True)
class _RequestFingerprint:
    kind: str
    start: int
    duration: int
    latest_end: int | None
    requirements: tuple[ResourceRequirement, ...]
    buffer_before: int
    buffer_after: int


@dataclass(frozen=True)
class IdempotencyEntry:
    """Checkpointed identity for an owner-scoped idempotent request."""

    owner: str
    request_id: str
    fingerprint: _RequestFingerprint
    reservation_id: str


@dataclass(frozen=True)
class ReservationCheckpoint:
    """Exact immutable state needed to recreate a ledger."""

    resources: tuple[ResourceCapacity, ...]
    reservations: tuple[Reservation, ...]
    next_sequences: tuple[tuple[str, int], ...]
    idempotency: tuple[IdempotencyEntry, ...]


class ReservationLedger:
    """Thread-safe deterministic scheduler for named cumulative resources.

    ``reserve_exact`` and ``reserve_earliest`` may request several resources.
    Every involved resource is checked before a new immutable state is exposed,
    giving all-or-nothing in-memory behavior.  The lock cannot provide a
    transaction across processes, services, or external side effects.
    """

    def __init__(self, resources: Mapping[str, CapacityInput]) -> None:
        if not isinstance(resources, Mapping):
            raise TypeError("resources must be a mapping")
        normalized: dict[str, CapacityVector] = {}
        for resource, capacity in resources.items():
            _non_empty_text(resource, "resource")
            normalized[resource] = self._as_capacity(capacity)
        if not normalized:
            raise ValueError("at least one resource is required")
        self._resources = dict(sorted(normalized.items()))
        self._reservations: dict[str, Reservation] = {}
        self._next_by_owner: dict[str, int] = {}
        self._idempotency: dict[tuple[str, str], tuple[_RequestFingerprint, str]] = {}
        self._lock = RLock()

    @staticmethod
    def _as_capacity(capacity: CapacityInput) -> CapacityVector:
        if isinstance(capacity, CapacityVector):
            return capacity
        if not isinstance(capacity, Mapping):
            raise TypeError("resource capacities must be CapacityVector or mappings")
        return CapacityVector(capacity)

    def _requirements(self, requirements: RequirementInput) -> tuple[ResourceRequirement, ...]:
        if not isinstance(requirements, Mapping):
            raise TypeError("requirements must be a resource mapping")
        normalized: list[ResourceRequirement] = []
        for resource, raw_capacity in requirements.items():
            _non_empty_text(resource, "resource")
            capacity = self._as_capacity(raw_capacity)
            available = self._resources.get(resource)
            if available is None:
                raise KeyError(f"unknown resource: {resource!r}")
            available._require_same_dimensions(capacity)
            if not available.fits(capacity):
                raise ValueError(
                    f"request exceeds total capacity of resource {resource!r}"
                )
            normalized.append(ResourceRequirement(resource, capacity))
        if not normalized:
            raise ValueError("at least one resource requirement is required")
        return tuple(sorted(normalized, key=lambda item: item.resource))

    def reserve_exact(
        self,
        owner: str,
        start: int,
        end: int,
        requirements: RequirementInput,
        *,
        request_id: str | None = None,
        buffer_before: int = 0,
        buffer_after: int = 0,
    ) -> Reservation:
        """Atomically reserve the exact half-open service interval ``[start, end)``."""
        _non_empty_text(owner, "owner")
        span = Span(start, end)
        if request_id is not None:
            _non_empty_text(request_id, "request_id")
        _non_negative_integer(buffer_before, "buffer_before")
        _non_negative_integer(buffer_after, "buffer_after")
        with self._lock:
            normalized = self._requirements(requirements)
            fingerprint = _RequestFingerprint(
                "exact",
                span.start,
                span.length,
                None,
                normalized,
                buffer_before,
                buffer_after,
            )
            replay = self._idempotent_replay(owner, request_id, fingerprint)
            if replay is not None:
                return replay
            return self._stage_and_commit(
                owner,
                span.start,
                span.end,
                normalized,
                request_id,
                fingerprint,
                buffer_before,
                buffer_after,
            )

    def reserve(
        self,
        owner: str,
        start: int,
        end: int,
        requirements: RequirementInput,
        *,
        request_id: str | None = None,
        buffer_before: int = 0,
        buffer_after: int = 0,
    ) -> Reservation:
        """Alias for :meth:`reserve_exact`."""
        return self.reserve_exact(
            owner,
            start,
            end,
            requirements,
            request_id=request_id,
            buffer_before=buffer_before,
            buffer_after=buffer_after,
        )

    def reserve_earliest(
        self,
        owner: str,
        duration: int,
        requirements: RequirementInput,
        *,
        earliest_start: int = 0,
        latest_end: int | None = None,
        request_id: str | None = None,
        buffer_before: int = 0,
        buffer_after: int = 0,
    ) -> Reservation:
        """Reserve the earliest feasible service interval in deterministic order.

        ``latest_end`` bounds the end of the service interval; buffers may extend
        outside that search window because they model surrounding occupancy.
        """
        _non_empty_text(owner, "owner")
        validate_length(duration)
        validate_coordinate(earliest_start, "earliest_start")
        if latest_end is not None:
            validate_coordinate(latest_end, "latest_end")
            if earliest_start + duration > latest_end:
                raise ValueError("duration does not fit within the search window")
        if request_id is not None:
            _non_empty_text(request_id, "request_id")
        _non_negative_integer(buffer_before, "buffer_before")
        _non_negative_integer(buffer_after, "buffer_after")

        with self._lock:
            normalized = self._requirements(requirements)
            fingerprint = _RequestFingerprint(
                "earliest",
                earliest_start,
                duration,
                latest_end,
                normalized,
                buffer_before,
                buffer_after,
            )
            replay = self._idempotent_replay(owner, request_id, fingerprint)
            if replay is not None:
                return replay

            latest_start = None if latest_end is None else latest_end - duration
            candidates = {earliest_start}
            requested_resources = {item.resource for item in normalized}
            for reservation in self._reservations.values():
                if not reservation.active:
                    continue
                if requested_resources.isdisjoint(
                    item.resource for item in reservation.requirements
                ):
                    continue
                candidate = reservation.occupied_span.end + buffer_before
                if candidate >= earliest_start:
                    candidates.add(candidate)

            last_conflicts: tuple[ReservationConflict, ...] = ()
            for start in sorted(candidates):
                if latest_start is not None and start > latest_start:
                    break
                end = start + duration
                conflicts = self._find_conflicts(
                    normalized,
                    start - buffer_before,
                    end + buffer_after,
                )
                if not conflicts:
                    return self._stage_and_commit(
                        owner,
                        start,
                        end,
                        normalized,
                        request_id,
                        fingerprint,
                        buffer_before,
                        buffer_after,
                        conflicts_checked=True,
                    )
                last_conflicts = conflicts

            if last_conflicts:
                raise ReservationConflictError(last_conflicts)
            raise ValueError("no reservation fits within the search window")

    def _idempotent_replay(
        self,
        owner: str,
        request_id: str | None,
        fingerprint: _RequestFingerprint,
    ) -> Reservation | None:
        if request_id is None:
            return None
        existing = self._idempotency.get((owner, request_id))
        if existing is None:
            return None
        previous, reservation_id = existing
        if previous != fingerprint:
            raise ValueError(
                "idempotency key was already used for a different reservation request"
            )
        return self._reservations[reservation_id]

    def _stage_and_commit(
        self,
        owner: str,
        start: int,
        end: int,
        requirements: tuple[ResourceRequirement, ...],
        request_id: str | None,
        fingerprint: _RequestFingerprint,
        buffer_before: int,
        buffer_after: int,
        *,
        conflicts_checked: bool = False,
    ) -> Reservation:
        occupied_start = start - buffer_before
        occupied_end = end + buffer_after
        if not conflicts_checked:
            conflicts = self._find_conflicts(
                requirements, occupied_start, occupied_end
            )
            if conflicts:
                raise ReservationConflictError(conflicts)

        sequence = self._next_by_owner.get(owner, 1)
        reservation_id = f"{owner}:{sequence}"
        if reservation_id in self._reservations:
            raise RuntimeError("reservation identity collision")
        staged = Reservation(
            reservation_id,
            owner,
            start,
            end,
            requirements,
            buffer_before,
            buffer_after,
            request_id,
        )

        # No state is changed until every resource and the complete record have
        # been validated.  These assignments are performed while holding the
        # ledger lock, so a partial multi-resource record is never observable.
        self._reservations[reservation_id] = staged
        self._next_by_owner[owner] = sequence + 1
        if request_id is not None:
            self._idempotency[(owner, request_id)] = (fingerprint, reservation_id)
        return staged

    def _find_conflicts(
        self,
        requirements: tuple[ResourceRequirement, ...],
        occupied_start: int,
        occupied_end: int,
        *,
        records: Mapping[str, Reservation] | None = None,
    ) -> tuple[ReservationConflict, ...]:
        source = self._reservations if records is None else records
        conflicts: list[ReservationConflict] = []
        for requirement in requirements:
            overlapping: list[tuple[Reservation, CapacityVector]] = []
            breakpoints = {occupied_start, occupied_end}
            for reservation in source.values():
                if not reservation.active:
                    continue
                existing = next(
                    (
                        item.capacity
                        for item in reservation.requirements
                        if item.resource == requirement.resource
                    ),
                    None,
                )
                if existing is None:
                    continue
                occupied = reservation.occupied_span
                if occupied.start < occupied_end and occupied_start < occupied.end:
                    overlapping.append((reservation, existing))
                    breakpoints.add(max(occupied_start, occupied.start))
                    breakpoints.add(min(occupied_end, occupied.end))

            points = sorted(breakpoints)
            available = self._resources[requirement.resource]
            zero = CapacityVector((name, 0) for name in available.dimensions)
            for segment_start, segment_end in zip(points, points[1:]):
                if segment_start == segment_end:
                    continue
                used = zero
                ids: list[str] = []
                for reservation, amount in overlapping:
                    span = reservation.occupied_span
                    if span.start < segment_end and segment_start < span.end:
                        used = used.add(amount)
                        ids.append(reservation.reservation_id)
                total = used.add(requirement.capacity)
                if available.fits(total):
                    continue
                detail = ReservationConflict(
                    requirement.resource,
                    segment_start,
                    segment_end,
                    available,
                    used,
                    requirement.capacity,
                    tuple(sorted(ids)),
                )
                if conflicts and self._can_merge_conflict(conflicts[-1], detail):
                    previous = conflicts[-1]
                    conflicts[-1] = replace(previous, end=detail.end)
                else:
                    conflicts.append(detail)
        return tuple(conflicts)

    @staticmethod
    def _can_merge_conflict(
        left: ReservationConflict, right: ReservationConflict
    ) -> bool:
        return (
            left.resource == right.resource
            and left.end == right.start
            and left.capacity == right.capacity
            and left.used == right.used
            and left.requested == right.requested
            and left.reservation_ids == right.reservation_ids
        )

    def cancel(self, owner: str, reservation_id: str) -> Reservation:
        """Cancel an owned reservation; repeated cancellation is idempotent."""
        _non_empty_text(owner, "owner")
        _non_empty_text(reservation_id, "reservation_id")
        with self._lock:
            reservation = self._reservations.get(reservation_id)
            if reservation is None:
                raise KeyError(reservation_id)
            if reservation.owner != owner:
                raise PermissionError("reservation belongs to a different owner")
            if not reservation.active:
                return reservation
            cancelled = replace(reservation, status=ReservationStatus.CANCELLED)
            self._reservations[reservation_id] = cancelled
            return cancelled

    def get(self, reservation_id: str) -> Reservation:
        """Return an immutable reservation by ID."""
        _non_empty_text(reservation_id, "reservation_id")
        with self._lock:
            try:
                return self._reservations[reservation_id]
            except KeyError:
                raise KeyError(reservation_id) from None

    def reservations(
        self, *, owner: str | None = None, active_only: bool = False
    ) -> tuple[Reservation, ...]:
        """Return reservations in deterministic time/identity order."""
        if owner is not None:
            _non_empty_text(owner, "owner")
        with self._lock:
            selected = (
                reservation
                for reservation in self._reservations.values()
                if (owner is None or reservation.owner == owner)
                and (not active_only or reservation.active)
            )
            return tuple(
                sorted(
                    selected,
                    key=lambda item: (item.start, item.end, item.reservation_id),
                )
            )

    def conflicts_for(
        self,
        start: int,
        end: int,
        requirements: RequirementInput,
        *,
        buffer_before: int = 0,
        buffer_after: int = 0,
    ) -> tuple[ReservationConflict, ...]:
        """Return deterministic conflict details without mutating the ledger."""
        span = Span(start, end)
        _non_negative_integer(buffer_before, "buffer_before")
        _non_negative_integer(buffer_after, "buffer_after")
        with self._lock:
            normalized = self._requirements(requirements)
            return self._find_conflicts(
                normalized,
                span.start - buffer_before,
                span.end + buffer_after,
            )

    def snapshot(self) -> ReservationSnapshot:
        """Return an immutable, deterministically ordered observable snapshot."""
        with self._lock:
            resources = tuple(
                ResourceCapacity(name, capacity)
                for name, capacity in self._resources.items()
            )
            return ReservationSnapshot(resources, self.reservations())

    def checkpoint(self) -> ReservationCheckpoint:
        """Return exact state, including counters and idempotency identities."""
        with self._lock:
            resources = tuple(
                ResourceCapacity(name, capacity)
                for name, capacity in self._resources.items()
            )
            reservations = tuple(
                sorted(self._reservations.values(), key=lambda item: item.reservation_id)
            )
            next_sequences = tuple(sorted(self._next_by_owner.items()))
            idempotency = tuple(
                IdempotencyEntry(owner, request_id, fingerprint, reservation_id)
                for (owner, request_id), (fingerprint, reservation_id) in sorted(
                    self._idempotency.items()
                )
            )
            return ReservationCheckpoint(
                resources, reservations, next_sequences, idempotency
            )

    @staticmethod
    def _validate_fingerprint(
        fingerprint: _RequestFingerprint, reservation: Reservation
    ) -> None:
        if not isinstance(fingerprint, _RequestFingerprint):
            raise TypeError("checkpoint idempotency fingerprint has an invalid type")
        if fingerprint.kind not in {"exact", "earliest"}:
            raise ValueError("checkpoint has an invalid request kind")
        validate_coordinate(fingerprint.start, "fingerprint start")
        validate_length(fingerprint.duration)
        if fingerprint.latest_end is not None:
            validate_coordinate(fingerprint.latest_end, "fingerprint latest_end")
        _non_negative_integer(fingerprint.buffer_before, "buffer_before")
        _non_negative_integer(fingerprint.buffer_after, "buffer_after")
        if (
            fingerprint.duration != reservation.span.length
            or fingerprint.requirements != reservation.requirements
            or fingerprint.buffer_before != reservation.buffer_before
            or fingerprint.buffer_after != reservation.buffer_after
        ):
            raise ValueError("idempotency fingerprint does not match its reservation")
        if fingerprint.kind == "exact":
            if (
                fingerprint.start != reservation.start
                or fingerprint.latest_end is not None
            ):
                raise ValueError("exact request fingerprint is inconsistent")
        elif (
            fingerprint.start > reservation.start
            or (
                fingerprint.latest_end is not None
                and reservation.end > fingerprint.latest_end
            )
        ):
            raise ValueError("earliest request fingerprint is inconsistent")

    @classmethod
    def from_checkpoint(cls, checkpoint: ReservationCheckpoint) -> ReservationLedger:
        """Build a ledger after validating a complete checkpoint atomically."""
        if not isinstance(checkpoint, ReservationCheckpoint):
            raise TypeError("checkpoint must be a ReservationCheckpoint")
        resource_names = tuple(item.resource for item in checkpoint.resources)
        if resource_names != tuple(sorted(resource_names)) or len(resource_names) != len(
            set(resource_names)
        ):
            raise ValueError("checkpoint resources must be unique and sorted")
        ledger = cls(
            {item.resource: item.capacity for item in checkpoint.resources}
        )

        records: dict[str, Reservation] = {}
        request_identities: set[tuple[str, str]] = set()
        owner_sequences: dict[str, list[int]] = {}
        for reservation in checkpoint.reservations:
            if reservation.reservation_id in records:
                raise ValueError("checkpoint contains duplicate reservation IDs")
            if reservation.request_id is not None:
                identity = (reservation.owner, reservation.request_id)
                if identity in request_identities:
                    raise ValueError(
                        "checkpoint contains a duplicate owner/request identity"
                    )
                request_identities.add(identity)
            sequence = _owner_sequence(
                reservation.reservation_id, reservation.owner
            )
            owner_sequences.setdefault(reservation.owner, []).append(sequence)
            normalized = ledger._requirements(
                {
                    item.resource: item.capacity
                    for item in reservation.requirements
                }
            )
            if normalized != reservation.requirements:
                raise ValueError("reservation requirements are not canonical")
            if reservation.active:
                conflicts = ledger._find_conflicts(
                    normalized,
                    reservation.occupied_span.start,
                    reservation.occupied_span.end,
                    records=records,
                )
                if conflicts:
                    raise ValueError("checkpoint exceeds resource capacity")
            records[reservation.reservation_id] = reservation

        next_by_owner: dict[str, int] = {}
        for owner, sequence in checkpoint.next_sequences:
            _non_empty_text(owner, "owner")
            if owner in next_by_owner:
                raise ValueError("checkpoint contains duplicate owner counters")
            validate_coordinate(sequence, "next sequence")
            if sequence <= 0:
                raise ValueError("next sequence must be positive")
            next_by_owner[owner] = sequence
        for owner in sorted(next_by_owner.keys() - owner_sequences.keys()):
            raise ValueError(f"checkpoint contains an orphan counter for {owner!r}")
        for owner, sequences in owner_sequences.items():
            next_sequence = next_by_owner.get(owner)
            if next_sequence is None:
                raise ValueError(f"checkpoint omits the owner counter for {owner!r}")
            if next_sequence <= max(sequences):
                raise ValueError("checkpoint owner counter would reuse an identity")
            if (
                len(sequences) != next_sequence - 1
                or sorted(sequences) != list(range(1, next_sequence))
            ):
                raise ValueError(
                    "checkpoint owner sequences are not contiguous through "
                    "the exact next counter"
                )

        idempotency: dict[
            tuple[str, str], tuple[_RequestFingerprint, str]
        ] = {}
        for entry in checkpoint.idempotency:
            _non_empty_text(entry.owner, "owner")
            _non_empty_text(entry.request_id, "request_id")
            key = (entry.owner, entry.request_id)
            if key in idempotency:
                raise ValueError("checkpoint contains duplicate idempotency keys")
            referenced = records.get(entry.reservation_id)
            if referenced is None:
                raise ValueError("idempotency entry references an unknown reservation")
            if referenced.owner != entry.owner or referenced.request_id != entry.request_id:
                raise ValueError("idempotency entry does not match its reservation")
            ledger._validate_fingerprint(entry.fingerprint, referenced)
            idempotency[key] = (entry.fingerprint, entry.reservation_id)
        for reservation in records.values():
            if reservation.request_id is None:
                continue
            identity = (reservation.owner, reservation.request_id)
            mapped = idempotency.get(identity)
            if mapped is None:
                raise ValueError("checkpoint omits a reservation idempotency identity")
            fingerprint, reservation_id = mapped
            if reservation_id != reservation.reservation_id:
                raise ValueError(
                    "checkpoint idempotency identity references a different reservation"
                )
            ledger._validate_fingerprint(fingerprint, reservation)

        ledger._reservations = records
        ledger._next_by_owner = next_by_owner
        ledger._idempotency = idempotency
        errors = ledger.diagnostics()
        if errors:
            raise ValueError(f"invalid reservation checkpoint: {errors[0]}")
        return ledger

    def restore(self, checkpoint: ReservationCheckpoint) -> None:
        """Atomically replace this ledger with validated checkpoint state."""
        restored = self.from_checkpoint(checkpoint)
        with self._lock:
            self._resources = restored._resources
            self._reservations = restored._reservations
            self._next_by_owner = restored._next_by_owner
            self._idempotency = restored._idempotency

    def diagnostics(self) -> tuple[str, ...]:
        """Return deterministic invariant violations; an empty tuple is healthy."""
        with self._lock:
            errors: list[str] = []
            staged: dict[str, Reservation] = {}
            owner_sequences: dict[str, list[int]] = {}
            request_identities: set[tuple[str, str]] = set()
            ordered_reservations = sorted(
                self._reservations.values(), key=lambda item: item.reservation_id
            )
            for reservation in ordered_reservations:
                if reservation.active:
                    conflicts = self._find_conflicts(
                        reservation.requirements,
                        reservation.occupied_span.start,
                        reservation.occupied_span.end,
                        records=staged,
                    )
                    if conflicts:
                        errors.append(
                            f"capacity exceeded by {reservation.reservation_id!r}"
                        )
                staged[reservation.reservation_id] = reservation
                try:
                    sequence = _owner_sequence(
                        reservation.reservation_id, reservation.owner
                    )
                except ValueError:
                    errors.append(
                        f"invalid identity {reservation.reservation_id!r}"
                    )
                else:
                    owner_sequences.setdefault(reservation.owner, []).append(sequence)
                if reservation.request_id is not None:
                    identity = (reservation.owner, reservation.request_id)
                    if identity in request_identities:
                        errors.append(
                            f"duplicate owner/request identity {identity!r}"
                        )
                    else:
                        request_identities.add(identity)

            for owner in sorted(self._next_by_owner.keys() - owner_sequences.keys()):
                errors.append(f"orphan owner counter for {owner!r}")
            for owner, sequences in sorted(owner_sequences.items()):
                next_sequence = self._next_by_owner.get(owner)
                if next_sequence is None:
                    errors.append(f"missing owner counter for {owner!r}")
                elif next_sequence <= max(sequences):
                    errors.append(f"stale owner counter for {owner!r}")
                elif (
                    len(sequences) != next_sequence - 1
                    or sorted(sequences) != list(range(1, next_sequence))
                ):
                    errors.append(
                        f"non-contiguous owner sequences or non-exact counter "
                        f"for {owner!r}"
                    )

            for (owner, request_id), (_, reservation_id) in sorted(
                self._idempotency.items()
            ):
                referenced = self._reservations.get(reservation_id)
                if referenced is None:
                    errors.append(
                        f"idempotency key {(owner, request_id)!r} has no reservation"
                    )
                elif (
                    referenced.owner != owner or referenced.request_id != request_id
                ):
                    errors.append(
                        f"idempotency key {(owner, request_id)!r} is inconsistent"
                    )

            for reservation in ordered_reservations:
                if reservation.request_id is None:
                    continue
                identity = (reservation.owner, reservation.request_id)
                mapped = self._idempotency.get(identity)
                if mapped is None:
                    errors.append(
                        f"reservation {reservation.reservation_id!r} has no "
                        "idempotency identity"
                    )
                    continue
                fingerprint, reservation_id = mapped
                if reservation_id != reservation.reservation_id:
                    errors.append(
                        f"reservation {reservation.reservation_id!r} has an "
                        "inconsistent idempotency identity"
                    )
                    continue
                try:
                    self._validate_fingerprint(fingerprint, reservation)
                except (TypeError, ValueError):
                    errors.append(
                        f"reservation {reservation.reservation_id!r} has an "
                        "inconsistent idempotency fingerprint"
                    )
            return tuple(errors)

    def assert_invariants(self) -> None:
        """Raise if internal cumulative-capacity invariants are violated."""
        errors = self.diagnostics()
        if errors:
            raise RuntimeError("; ".join(errors))
