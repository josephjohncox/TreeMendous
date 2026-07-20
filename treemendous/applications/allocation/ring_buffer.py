"""Bounded ring-buffer capacity over modular producer sequences."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock

from treemendous.applications._shared.ring_sequences import (
    RingSequenceCheckpoint,
    RingSequenceSnapshot,
    RingSequenceTracker,
)
from treemendous.domain import Span, validate_coordinate, validate_length


class RingFullError(RuntimeError):
    """Raised when backpressure prevents a producer reservation."""


class RingEmptyError(RuntimeError):
    """Raised when a consumer requests unavailable entries."""


class FullPolicy(str, Enum):
    """Behavior when production exceeds currently free slots."""

    BACKPRESSURE = "backpressure"
    OVERWRITE = "overwrite"


@dataclass(frozen=True)
class ProducedRange:
    """One producer reservation in unwrapped sequence space."""

    sequences: Span
    modular_start: int
    start_epoch: int
    overwritten: int


@dataclass(frozen=True)
class ConsumedRange:
    """One consumer advancement in unwrapped sequence space."""

    sequences: Span
    modular_start: int
    start_epoch: int


@dataclass(frozen=True)
class RingBufferSnapshot:
    """Immutable cursor, occupancy, and modular tracking state."""

    capacity: int
    modulus: int
    policy: FullPolicy
    producer_cursor: int
    consumer_cursor: int
    occupancy: int
    free_slots: int
    overwritten: int
    sequences: RingSequenceSnapshot


@dataclass(frozen=True)
class RingBufferCheckpoint:
    """Restorable ring buffer cursors and modular sequence state."""

    producer_cursor: int
    consumer_cursor: int
    overwritten: int
    sequences: RingSequenceCheckpoint


class RingBuffer:
    """Reserve/consume ring capacity with explicit wrap epoch validation.

    The tracker records every produced modular sequence.  At most ``capacity``
    entries are outstanding, so modular identities are unique while resident.
    Backpressure rejects overflow atomically; overwrite advances the consumer
    cursor and reports exactly how many unread entries were lost.
    """

    def __init__(
        self,
        capacity: int,
        *,
        sequence_modulus: int = 2**32,
        initial_sequence: int = 0,
        full_policy: FullPolicy | str = FullPolicy.BACKPRESSURE,
    ) -> None:
        validate_length(capacity)
        validate_coordinate(sequence_modulus, "sequence_modulus")
        if sequence_modulus < 2 or capacity > sequence_modulus:
            raise ValueError("sequence_modulus must be at least capacity and two")
        validate_coordinate(initial_sequence, "initial_sequence")
        if not 0 <= initial_sequence < sequence_modulus:
            raise ValueError("initial_sequence must be in the modular domain")
        try:
            self._policy = FullPolicy(full_policy)
        except (TypeError, ValueError) as error:
            raise ValueError("full_policy must be backpressure or overwrite") from error
        self._capacity = capacity
        self._modulus = sequence_modulus
        self._initial = initial_sequence
        self._producer = initial_sequence
        self._consumer = initial_sequence
        self._overwritten = 0
        self._tracker = RingSequenceTracker(
            sequence_modulus, initial_sequence=initial_sequence, max_ranges=2
        )
        self._lock = RLock()

    def produce(self, count: int, *, epoch_hint: int | None = None) -> ProducedRange:
        """Reserve ``count`` consecutive producer sequences."""
        validate_length(count)
        if epoch_hint is not None:
            validate_coordinate(epoch_hint, "epoch_hint")
            if epoch_hint < 0:
                raise ValueError("epoch_hint must be nonnegative")
        with self._lock:
            expected_epoch = self._producer // self._modulus
            if epoch_hint is not None and epoch_hint != expected_epoch:
                raise ValueError("epoch_hint does not identify the producer cursor")
            occupancy = self._producer - self._consumer
            overflow = max(0, occupancy + count - self._capacity)
            if overflow and self._policy is FullPolicy.BACKPRESSURE:
                raise RingFullError("ring buffer has insufficient free capacity")
            tracker_checkpoint = self._tracker.checkpoint()
            start = self._producer
            try:
                for absolute in range(start, start + count):
                    self._tracker.observe(
                        absolute % self._modulus, epoch=absolute // self._modulus
                    )
            except Exception:
                self._tracker.restore(tracker_checkpoint)
                raise
            self._producer += count
            if overflow:
                self._consumer += overflow
                self._overwritten += overflow
            return ProducedRange(
                sequences=Span(start, self._producer),
                modular_start=start % self._modulus,
                start_epoch=expected_epoch,
                overwritten=overflow,
            )

    def consume(self, count: int) -> ConsumedRange:
        """Advance the consumer cursor over available entries."""
        validate_length(count)
        with self._lock:
            if count > self._producer - self._consumer:
                raise RingEmptyError("consumer requested more entries than available")
            start = self._consumer
            self._consumer += count
            return ConsumedRange(
                sequences=Span(start, self._consumer),
                modular_start=start % self._modulus,
                start_epoch=start // self._modulus,
            )

    def snapshot(self) -> RingBufferSnapshot:
        """Return cursor, occupancy, overwrite, and sequence diagnostics."""
        with self._lock:
            occupancy = self._producer - self._consumer
            return RingBufferSnapshot(
                capacity=self._capacity,
                modulus=self._modulus,
                policy=self._policy,
                producer_cursor=self._producer,
                consumer_cursor=self._consumer,
                occupancy=occupancy,
                free_slots=self._capacity - occupancy,
                overwritten=self._overwritten,
                sequences=self._tracker.snapshot(),
            )

    def checkpoint(self) -> RingBufferCheckpoint:
        """Capture complete ring-buffer state."""
        with self._lock:
            return RingBufferCheckpoint(
                self._producer,
                self._consumer,
                self._overwritten,
                self._tracker.checkpoint(),
            )

    def restore(self, checkpoint: RingBufferCheckpoint) -> None:
        """Atomically restore a structurally valid local checkpoint."""
        if not isinstance(checkpoint, RingBufferCheckpoint):
            raise TypeError("checkpoint must be a RingBufferCheckpoint")
        for value, name in (
            (checkpoint.producer_cursor, "producer_cursor"),
            (checkpoint.consumer_cursor, "consumer_cursor"),
            (checkpoint.overwritten, "overwritten"),
        ):
            validate_coordinate(value, name)
        if (
            checkpoint.consumer_cursor < self._initial
            or checkpoint.producer_cursor < checkpoint.consumer_cursor
            or checkpoint.producer_cursor - checkpoint.consumer_cursor > self._capacity
            or checkpoint.overwritten < 0
            or checkpoint.overwritten > checkpoint.consumer_cursor - self._initial
            or (self._policy is FullPolicy.BACKPRESSURE and checkpoint.overwritten != 0)
        ):
            raise ValueError("checkpoint contains invalid ring cursors")
        expected_highest = (
            None
            if checkpoint.producer_cursor == self._initial
            else checkpoint.producer_cursor - 1
        )
        expected_reference = (
            self._initial if expected_highest is None else expected_highest
        )
        if (
            checkpoint.sequences.origin != self._initial
            or checkpoint.sequences.reference != expected_reference
            or checkpoint.sequences.highest_received != expected_highest
        ):
            raise ValueError(
                "checkpoint tracker origin and high-water state do not match cursors"
            )
        ranges = checkpoint.sequences.received_ranges
        expected = (
            ()
            if checkpoint.producer_cursor == self._initial
            else (Span(self._initial, checkpoint.producer_cursor),)
        )
        if ranges != expected:
            raise ValueError(
                "checkpoint sequence history does not match producer cursor"
            )
        with self._lock:
            self._tracker.restore(checkpoint.sequences)
            self._producer = checkpoint.producer_cursor
            self._consumer = checkpoint.consumer_cursor
            self._overwritten = checkpoint.overwritten


def create_application(**kwargs: object) -> RingBuffer:
    """Registry factory for :class:`RingBuffer`."""
    return RingBuffer(**kwargs)  # type: ignore[arg-type]
