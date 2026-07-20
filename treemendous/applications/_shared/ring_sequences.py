"""Bounded modular sequence unwrapping and gap tracking."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from treemendous.domain import Span, validate_coordinate


class RingSequenceError(Exception):
    """Base error for cyclic sequence tracking."""


class AmbiguousSequenceError(RingSequenceError):
    """Raised when two epochs are equally close at half the modulus."""


class SequenceBeforeOriginError(RingSequenceError):
    """Raised when an observation unwraps before the tracking origin."""


@dataclass(frozen=True)
class SequenceObservation:
    """Result of recording one bounded sequence number."""

    sequence: int
    unwrapped: int
    epoch: int
    duplicate: bool
    contiguous_range: Span | None


@dataclass(frozen=True)
class RingSequenceSnapshot:
    """Immutable received, contiguous, and missing sequence geometry."""

    modulus: int
    origin: int | None
    reference: int | None
    received_ranges: tuple[Span, ...]
    contiguous_range: Span | None
    missing_ranges: tuple[Span, ...]


@dataclass(frozen=True)
class RingSequenceCheckpoint:
    """Complete state for restoring a ring sequence tracker."""

    modulus: int
    origin: int | None
    reference: int | None
    highest_received: int | None
    received_ranges: tuple[Span, ...]


class RingSequenceTracker:
    """Unwrap and track modular sequence numbers using a nearest-epoch rule.

    Each observation must be within strictly less than half a modulus of the
    current high-water reference.  Modular values exactly half a modulus away
    are rejected because past and future epochs are equally plausible.  If
    ``initial_sequence`` is omitted, the first observation establishes both
    the origin and epoch zero.
    """

    def __init__(self, modulus: int, *, initial_sequence: int | None = None) -> None:
        validate_coordinate(modulus, "modulus")
        if modulus < 2:
            raise ValueError("modulus must be at least two")
        if initial_sequence is not None:
            self._validate_sequence_for_modulus(initial_sequence, modulus)

        self._modulus = modulus
        self._origin = initial_sequence
        self._reference = initial_sequence
        self._highest_received: int | None = None
        self._received: tuple[Span, ...] = ()
        self._lock = RLock()

    @property
    def modulus(self) -> int:
        return self._modulus

    @property
    def origin(self) -> int | None:
        with self._lock:
            return self._origin

    @property
    def reference(self) -> int | None:
        with self._lock:
            return self._reference

    @property
    def received_ranges(self) -> tuple[Span, ...]:
        with self._lock:
            return self._received

    @property
    def contiguous_range(self) -> Span | None:
        with self._lock:
            return self._contiguous_range()

    @property
    def missing_ranges(self) -> tuple[Span, ...]:
        with self._lock:
            return self._missing_ranges()

    def unwrap(self, sequence: int) -> int:
        """Return the uniquely nearest unwrapped value without recording it."""
        self._validate_sequence(sequence)
        with self._lock:
            return self._unwrap(sequence)

    def observe(self, sequence: int) -> SequenceObservation:
        """Record a sequence number and return its unwrapped epoch information."""
        self._validate_sequence(sequence)
        with self._lock:
            unwrapped = self._unwrap(sequence)
            origin = self._origin if self._origin is not None else unwrapped
            if unwrapped < origin:
                raise SequenceBeforeOriginError(
                    "sequence unwraps before the tracking origin"
                )

            received, duplicate = self._with_point(self._received, unwrapped)
            highest = (
                unwrapped
                if self._highest_received is None
                else max(self._highest_received, unwrapped)
            )
            reference = max(origin, highest)

            # Commit only after all validation and derived-state construction.
            self._origin = origin
            self._received = received
            self._highest_received = highest
            self._reference = reference
            return SequenceObservation(
                sequence=sequence,
                unwrapped=unwrapped,
                epoch=unwrapped // self._modulus,
                duplicate=duplicate,
                contiguous_range=self._contiguous_range(),
            )

    def snapshot(self) -> RingSequenceSnapshot:
        """Return all tracked range views in one internally consistent value."""
        with self._lock:
            return RingSequenceSnapshot(
                modulus=self._modulus,
                origin=self._origin,
                reference=self._reference,
                received_ranges=self._received,
                contiguous_range=self._contiguous_range(),
                missing_ranges=self._missing_ranges(),
            )

    def checkpoint(self) -> RingSequenceCheckpoint:
        """Capture complete tracker state."""
        with self._lock:
            return RingSequenceCheckpoint(
                modulus=self._modulus,
                origin=self._origin,
                reference=self._reference,
                highest_received=self._highest_received,
                received_ranges=self._received,
            )

    def restore(self, checkpoint: RingSequenceCheckpoint) -> None:
        """Atomically restore a structurally valid checkpoint of this modulus."""
        if not isinstance(checkpoint, RingSequenceCheckpoint):
            raise TypeError("checkpoint must be a RingSequenceCheckpoint")
        with self._lock:
            self._validate_checkpoint(checkpoint)
            self._origin = checkpoint.origin
            self._reference = checkpoint.reference
            self._highest_received = checkpoint.highest_received
            self._received = checkpoint.received_ranges

    @staticmethod
    def _validate_sequence_for_modulus(sequence: int, modulus: int) -> None:
        validate_coordinate(sequence, "sequence")
        if not 0 <= sequence < modulus:
            raise ValueError("sequence must satisfy 0 <= sequence < modulus")

    def _validate_sequence(self, sequence: int) -> None:
        self._validate_sequence_for_modulus(sequence, self._modulus)

    def _unwrap(self, sequence: int) -> int:
        reference = self._reference
        if reference is None:
            return sequence
        reference_modular = reference % self._modulus
        forward = (sequence - reference_modular) % self._modulus
        if forward * 2 == self._modulus:
            raise AmbiguousSequenceError(
                "sequence jump is ambiguous at half the modulus"
            )
        delta = forward if forward * 2 < self._modulus else forward - self._modulus
        return reference + delta

    @staticmethod
    def _with_point(
        ranges: tuple[Span, ...], point: int
    ) -> tuple[tuple[Span, ...], bool]:
        addition = Span(point, point + 1)
        merged: list[Span] = []
        duplicate = False
        inserted = False
        for current in ranges:
            if current.start <= point < current.end:
                duplicate = True
            if current.end < addition.start:
                merged.append(current)
            elif addition.end < current.start:
                if not inserted:
                    merged.append(addition)
                    inserted = True
                merged.append(current)
            else:
                addition = Span(
                    min(addition.start, current.start),
                    max(addition.end, current.end),
                )
        if not inserted:
            merged.append(addition)
        return tuple(merged), duplicate

    def _contiguous_range(self) -> Span | None:
        if self._origin is None:
            return None
        cursor = self._origin
        for span in self._received:
            if span.start > cursor:
                break
            cursor = span.end
        return Span(self._origin, cursor) if cursor > self._origin else None

    def _missing_ranges(self) -> tuple[Span, ...]:
        if self._origin is None or self._highest_received is None:
            return ()
        cursor = self._origin
        missing: list[Span] = []
        for span in self._received:
            if cursor < span.start:
                missing.append(Span(cursor, span.start))
            cursor = span.end
        return tuple(missing)

    def _validate_checkpoint(self, checkpoint: RingSequenceCheckpoint) -> None:
        if checkpoint.modulus != self._modulus:
            raise ValueError("checkpoint modulus does not match tracker modulus")
        origin = checkpoint.origin
        reference = checkpoint.reference
        highest = checkpoint.highest_received
        ranges = checkpoint.received_ranges

        if origin is None:
            if reference is not None or highest is not None or ranges:
                raise ValueError("uninitialized checkpoint contains sequence state")
            return
        validate_coordinate(origin, "checkpoint origin")
        if not 0 <= origin < self._modulus:
            raise ValueError("checkpoint origin is outside the modular domain")
        if reference is None:
            raise ValueError("checkpoint reference is invalid")
        validate_coordinate(reference, "checkpoint reference")
        if reference < origin:
            raise ValueError("checkpoint reference is invalid")
        if highest is not None:
            validate_coordinate(highest, "checkpoint highest_received")
        if any(span.start < origin for span in ranges):
            raise ValueError("checkpoint range precedes its origin")
        for previous, current in zip(ranges, ranges[1:]):
            if previous.end >= current.start:
                raise ValueError("checkpoint ranges are not normalized")

        if not ranges:
            if highest is not None or reference != origin:
                raise ValueError("empty checkpoint has inconsistent high-water state")
            return
        expected_highest = ranges[-1].end - 1
        if highest != expected_highest or reference != max(origin, expected_highest):
            raise ValueError("checkpoint high-water state is inconsistent")
