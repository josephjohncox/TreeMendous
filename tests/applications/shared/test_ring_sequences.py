"""Contracts for bounded modular sequence unwrapping and gap tracking."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import pytest

from treemendous.applications._shared.ring_sequences import (
    AmbiguousSequenceError,
    RingSequenceTracker,
    SequenceBeforeOriginError,
)
from treemendous.domain import Span


def test_wraparound_unwraps_into_epochs_and_duplicates_are_idempotent() -> None:
    tracker = RingSequenceTracker(8, initial_sequence=6)

    assert tracker.observe(6).unwrapped == 6
    assert tracker.observe(7).unwrapped == 7
    wrapped = tracker.observe(0)
    duplicate = tracker.observe(7)

    assert wrapped.unwrapped == 8
    assert wrapped.epoch == 1
    assert duplicate.unwrapped == 7
    assert duplicate.duplicate
    assert tracker.contiguous_range == Span(6, 9)
    expected_received = (Span(6, 9),)
    assert tracker.received_ranges == expected_received
    assert not tracker.missing_ranges


def test_out_of_order_observations_track_missing_and_contiguous_ranges() -> None:
    tracker = RingSequenceTracker(16, initial_sequence=14)

    tracker.observe(14)
    tracker.observe(0)
    assert tracker.contiguous_range == Span(14, 15)
    expected_missing = (Span(15, 16),)
    assert tracker.missing_ranges == expected_missing

    filled = tracker.observe(15)
    assert filled.unwrapped == 15
    assert filled.contiguous_range == Span(14, 17)
    assert not tracker.missing_ranges


def test_first_observation_establishes_origin_when_unspecified() -> None:
    tracker = RingSequenceTracker(32)

    first = tracker.observe(29)
    wrapped = tracker.observe(0)

    assert first.unwrapped == 29
    assert tracker.origin == 29
    assert wrapped.unwrapped == 32
    assert wrapped.epoch == 1
    expected_received = (Span(29, 30), Span(32, 33))
    expected_missing = (Span(30, 32),)
    assert tracker.received_ranges == expected_received
    assert tracker.missing_ranges == expected_missing


def test_exact_half_modulus_jump_is_rejected_without_mutation() -> None:
    tracker = RingSequenceTracker(8, initial_sequence=0)
    before = tracker.snapshot()

    with pytest.raises(AmbiguousSequenceError, match="half"):
        tracker.observe(4)

    assert tracker.snapshot() == before


def test_sequence_that_unwraps_before_origin_is_rejected() -> None:
    tracker = RingSequenceTracker(8, initial_sequence=0)

    with pytest.raises(SequenceBeforeOriginError, match="before"):
        tracker.observe(7)
    assert not tracker.snapshot().received_ranges


def test_odd_modulus_has_no_exact_half_modulus_ambiguity() -> None:
    tracker = RingSequenceTracker(7, initial_sequence=0)

    assert tracker.observe(3).unwrapped == 3
    assert tracker.observe(4).unwrapped == 4


def test_unwrap_does_not_mutate_tracker() -> None:
    tracker = RingSequenceTracker(16, initial_sequence=14)
    before = tracker.snapshot()

    assert tracker.unwrap(1) == 17
    assert tracker.snapshot() == before


def test_checkpoint_restores_received_contiguous_and_missing_state() -> None:
    tracker = RingSequenceTracker(16, initial_sequence=14)
    tracker.observe(14)
    tracker.observe(0)
    checkpoint = tracker.checkpoint()
    tracker.observe(15)

    tracker.restore(checkpoint)

    expected_received = (Span(14, 15), Span(16, 17))
    expected_missing = (Span(15, 16),)
    assert tracker.received_ranges == expected_received
    assert tracker.contiguous_range == Span(14, 15)
    assert tracker.missing_ranges == expected_missing
    assert tracker.observe(15).contiguous_range == Span(14, 17)


def test_out_of_order_point_insertion_preserves_coordinate_order() -> None:
    tracker = RingSequenceTracker(16, initial_sequence=0)
    tracker.observe(2)
    tracker.observe(0)

    expected = (Span(0, 1), Span(2, 3))
    assert tracker.received_ranges == expected


def test_checkpoint_validation_is_failure_atomic() -> None:
    tracker = RingSequenceTracker(8, initial_sequence=1)
    tracker.observe(1)
    checkpoint = tracker.checkpoint()
    malformed = replace(checkpoint, highest_received=4)
    before = tracker.snapshot()

    with pytest.raises(ValueError, match="high-water"):
        tracker.restore(malformed)
    assert tracker.snapshot() == before


def test_restore_validates_checkpoint_structure_and_empty_states() -> None:
    tracker = RingSequenceTracker(8, initial_sequence=1)
    empty = tracker.checkpoint()
    tracker.observe(1)
    tracker.restore(empty)
    assert tracker.modulus == 8
    assert tracker.reference == 1
    assert not tracker.received_ranges

    before = tracker.snapshot()
    invalid = [
        replace(empty, modulus=16),
        replace(empty, origin=-1, reference=-1),
        replace(empty, reference=None),
        replace(empty, reference=0),
        replace(empty, highest_received=2),
        replace(
            empty,
            highest_received=1,
            received_ranges=(Span(0, 1),),
        ),
        replace(
            empty,
            reference=3,
            highest_received=3,
            received_ranges=(Span(1, 3), Span(2, 4)),
        ),
    ]
    for malformed in invalid:
        with pytest.raises((TypeError, ValueError)):
            tracker.restore(malformed)
        assert tracker.snapshot() == before

    uninitialized = RingSequenceTracker(8)
    pristine = uninitialized.checkpoint()
    uninitialized.restore(pristine)
    corrupted = replace(pristine, reference=0)
    with pytest.raises(ValueError, match="uninitialized"):
        uninitialized.restore(corrupted)

    invalid_checkpoint: Any = object()
    with pytest.raises(TypeError, match="RingSequenceCheckpoint"):
        tracker.restore(invalid_checkpoint)


def test_validation_rejects_invalid_modulus_and_sequence_numbers() -> None:
    with pytest.raises(ValueError, match="at least two"):
        RingSequenceTracker(1)
    with pytest.raises(TypeError, match="integer"):
        RingSequenceTracker(True)

    tracker = RingSequenceTracker(8)
    with pytest.raises(ValueError, match="0 <= sequence"):
        tracker.observe(8)
    with pytest.raises(ValueError, match="0 <= sequence"):
        RingSequenceTracker(8, initial_sequence=-1)


def test_concurrent_observations_preserve_normalized_received_ranges() -> None:
    tracker = RingSequenceTracker(512, initial_sequence=0)
    with ThreadPoolExecutor(max_workers=12) as executor:
        observations = list(executor.map(tracker.observe, range(100)))

    assert {item.unwrapped for item in observations} == set(range(100))
    expected_received = (Span(0, 100),)
    assert tracker.received_ranges == expected_received
    assert tracker.contiguous_range == Span(0, 100)
    assert not tracker.missing_ranges
