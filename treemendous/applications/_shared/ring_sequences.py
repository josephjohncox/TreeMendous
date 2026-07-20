"""Bounded modular sequence unwrapping and gap tracking."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from treemendous.domain import Span, validate_coordinate


class RingSequenceError(Exception):
    """Base error for cyclic sequence tracking."""


class AmbiguousSequenceError(RingSequenceError):
    """Raised when a modular value lacks enough information to select an epoch."""


class SequenceBeforeOriginError(RingSequenceError):
    """Raised when an observation unwraps before the tracking origin."""


class RangeBudgetExceededError(RingSequenceError):
    """Raised when an observation would exceed the received-range budget."""


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
    max_ranges: int
    origin: int | None
    reference: int | None
    received_ranges: tuple[Span, ...]
    contiguous_range: Span | None
    missing_ranges: tuple[Span, ...]


@dataclass(frozen=True)
class RingSequenceCheckpoint:
    """Complete state for restoring a ring sequence tracker."""

    modulus: int
    max_ranges: int
    origin: int | None
    reference: int | None
    highest_received: int | None
    received_ranges: tuple[Span, ...]


class RingSequenceTracker:
    """Unwrap modular values and retain bounded received-range geometry.

    Before the first wrap, observations use the nearest-epoch rule and reject
    exact half-modulus ambiguity.  After entering a nonzero epoch, callers must
    identify every observation's epoch explicitly so delayed old values cannot
    be silently reclassified.  ``max_ranges`` bounds retained fragmentation.
    If ``initial_sequence`` is omitted, the first observation establishes both
    the origin and epoch zero.
    """

    def __init__(
        self,
        modulus: int,
        *,
        initial_sequence: int | None = None,
        max_ranges: int = 1024,
    ) -> None:
        validate_coordinate(modulus, "modulus")
        if modulus < 2:
            raise ValueError("modulus must be at least two")
        self._validate_max_ranges(max_ranges)
        if initial_sequence is not None:
            self._validate_sequence_for_modulus(initial_sequence, modulus)

        self._modulus = modulus
        self._max_ranges = max_ranges
        self._origin = initial_sequence
        self._reference = initial_sequence
        self._highest_received: int | None = None
        self._received: tuple[Span, ...] = ()
        self._lock = RLock()

    @property
    def modulus(self) -> int:
        return self._modulus

    @property
    def max_ranges(self) -> int:
        return self._max_ranges

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

    def unwrap(self, sequence: int, *, epoch: int | None = None) -> int:
        """Return an unwrapped value without recording it.

        After the tracker enters a nonzero epoch, ``epoch`` is required so an
        old delayed value cannot be silently interpreted as a new packet.
        """
        self._validate_sequence(sequence)
        self._validate_epoch(epoch)
        with self._lock:
            return self._unwrap(sequence, epoch)

    def observe(
        self, sequence: int, *, epoch: int | None = None
    ) -> SequenceObservation:
        """Record a sequence number and return its unwrapped epoch information."""
        self._validate_sequence(sequence)
        self._validate_epoch(epoch)
        with self._lock:
            unwrapped = self._unwrap(sequence, epoch)
            origin = self._origin if self._origin is not None else unwrapped
            if unwrapped < origin:
                raise SequenceBeforeOriginError(
                    "sequence unwraps before the tracking origin"
                )

            received, duplicate = self._with_point(self._received, unwrapped)
            if len(received) > self._max_ranges:
                raise RangeBudgetExceededError(
                    f"received range budget of {self._max_ranges} would be exceeded"
                )
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
                max_ranges=self._max_ranges,
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
                max_ranges=self._max_ranges,
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
            received = self._validate_checkpoint(checkpoint)
            self._origin = checkpoint.origin
            self._reference = checkpoint.reference
            self._highest_received = checkpoint.highest_received
            self._received = received

    @staticmethod
    def _validate_max_ranges(max_ranges: int) -> None:
        validate_coordinate(max_ranges, "max_ranges")
        if max_ranges <= 0:
            raise ValueError("max_ranges must be greater than zero")

    @staticmethod
    def _validate_sequence_for_modulus(sequence: int, modulus: int) -> None:
        validate_coordinate(sequence, "sequence")
        if not 0 <= sequence < modulus:
            raise ValueError("sequence must satisfy 0 <= sequence < modulus")

    def _validate_sequence(self, sequence: int) -> None:
        self._validate_sequence_for_modulus(sequence, self._modulus)

    @staticmethod
    def _validate_epoch(epoch: int | None) -> None:
        if epoch is None:
            return
        validate_coordinate(epoch, "epoch")
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")

    def _unwrap(self, sequence: int, epoch: int | None) -> int:
        reference = self._reference
        if epoch is not None:
            return epoch * self._modulus + sequence
        if reference is None:
            return sequence
        if reference >= self._modulus:
            raise AmbiguousSequenceError(
                "epoch is required after the sequence has wrapped"
            )
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

    def _validate_checkpoint(
        self, checkpoint: RingSequenceCheckpoint
    ) -> tuple[Span, ...]:
        if checkpoint.modulus != self._modulus:
            raise ValueError("checkpoint modulus does not match tracker modulus")
        self._validate_max_ranges(checkpoint.max_ranges)
        if checkpoint.max_ranges != self._max_ranges:
            raise ValueError("checkpoint range budget does not match tracker policy")
        origin = checkpoint.origin
        reference = checkpoint.reference
        highest = checkpoint.highest_received
        try:
            ranges = tuple(span for span in checkpoint.received_ranges)
        except TypeError as error:
            raise TypeError("checkpoint received_ranges must be iterable") from error
        if any(not isinstance(span, Span) for span in ranges):
            raise TypeError("checkpoint received ranges must contain only Span values")
        if len(ranges) > self._max_ranges:
            raise ValueError("checkpoint exceeds the received range budget")

        if origin is None:
            if reference is not None or highest is not None or ranges:
                raise ValueError("uninitialized checkpoint contains sequence state")
            return ranges
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
            return ranges
        expected_highest = ranges[-1].end - 1
        if highest != expected_highest or reference != max(origin, expected_highest):
            raise ValueError("checkpoint high-water state is inconsistent")
        return ranges
