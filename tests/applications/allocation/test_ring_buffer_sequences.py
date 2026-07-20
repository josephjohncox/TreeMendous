"""Ring buffer application contracts."""

import pytest

from tests.oracles.applications.allocation.ring_buffer_sequences import produce
from treemendous.applications.allocation.ring_buffer import (
    FullPolicy,
    RingBuffer,
    RingFullError,
)


def test_wrap_epoch_hints_and_backpressure_are_atomic() -> None:
    ring = RingBuffer(4, sequence_modulus=8, initial_sequence=6)
    produced = ring.produce(2, epoch_hint=0)
    assert produced.modular_start == 6
    assert produced.start_epoch == 0
    wrapped = ring.produce(2, epoch_hint=1)
    assert wrapped.modular_start == 0
    assert wrapped.start_epoch == 1
    before = ring.snapshot()
    with pytest.raises(RingFullError):
        ring.produce(1)
    assert ring.snapshot() == before
    assert before.sequences.contiguous_range == produced.sequences.__class__(6, 10)


def test_overwrite_policy_oracle_consume_and_checkpoint() -> None:
    ring = RingBuffer(3, sequence_modulus=8, full_policy=FullPolicy.OVERWRITE)
    expected_producer, expected_consumer, expected_overwritten = produce(
        3, 0, 0, 5, True
    )
    result = ring.produce(5)
    snapshot = ring.snapshot()
    assert snapshot.producer_cursor == expected_producer
    assert snapshot.consumer_cursor == expected_consumer
    assert result.overwritten == expected_overwritten
    checkpoint = ring.checkpoint()
    consumed = ring.consume(2)
    assert consumed.sequences.start == 2
    ring.restore(checkpoint)
    assert ring.snapshot().occupancy == 3
