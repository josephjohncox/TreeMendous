"""Contracts for private deterministic event streams."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from math import nan
from typing import Any

import pytest

from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications._shared.events import (
    Event,
    EventIdempotencyConflictError,
    EventLog,
    ExpectedVersionError,
    InvalidEventCheckpointError,
    freeze_metadata,
)


def test_stream_and_global_versions_are_deterministic() -> None:
    clock = LogicalClock(10)
    log = EventLog(clock=clock)

    first = log.append("job:1", "created", {"items": [1, 2]}, expected_version=0)
    clock.advance()
    second = log.append("job:2", "created", expected_version=0)
    third = log.append("job:1", "finished", expected_version=1)

    observed_sequences = (first.sequence, second.sequence, third.sequence)
    observed_versions = (first.version, second.version, third.version)
    expected_sequences = (1, 2, 3)
    expected_versions = (1, 1, 2)
    expected_stream_events = (first, third)
    expected_stream_versions = (("job:1", 2), ("job:2", 1))
    assert observed_sequences == expected_sequences
    assert observed_versions == expected_versions
    assert third.occurred_at == 11
    assert log.events("job:1") == expected_stream_events
    assert log.version("missing") == 0
    assert log.snapshot().stream_versions == expected_stream_versions


def test_expected_version_and_invalid_payload_fail_atomically() -> None:
    log = EventLog(clock=LogicalClock())
    original = log.append("stream", "created")
    before = log.snapshot()

    with pytest.raises(ExpectedVersionError, match="version 1"):
        log.append("stream", "wrong", expected_version=0)
    with pytest.raises(TypeError, match="payload values"):
        log.append("stream", "bad", {"object": object()})

    expected_events = (original,)
    assert log.snapshot() == before
    assert log.events() == expected_events


def test_idempotency_returns_original_only_for_identical_request() -> None:
    clock = LogicalClock(3)
    log = EventLog(clock=clock)
    original = log.append(
        "stream",
        "created",
        {"nested": {"b": 2, "a": [1]}},
        expected_version=0,
        idempotency_key="request-1",
    )
    clock.advance(10)
    retry = log.append(
        "stream",
        "created",
        {"nested": {"a": [1], "b": 2}},
        expected_version=0,
        idempotency_key="request-1",
    )

    assert retry is original
    assert retry.occurred_at == 3
    assert len(log.events()) == 1
    with pytest.raises(EventIdempotencyConflictError):
        log.append(
            "stream",
            "different",
            expected_version=0,
            idempotency_key="request-1",
        )


def test_payload_is_recursively_immutable_and_detached() -> None:
    source = {"values": [1, {"key": "value"}]}
    event = EventLog(clock=LogicalClock()).append("stream", "kind", source)
    source["values"].append(3)

    expected_payload = freeze_metadata({"values": [1, {"key": "value"}]})
    assert event.payload == expected_payload
    with pytest.raises(Exception):
        event.payload[0] = ("changed", ())  # type: ignore[index]


def test_metadata_freezing_distinguishes_mappings_from_sequences() -> None:
    log = EventLog(clock=LogicalClock())
    original = log.append(
        "stream",
        "kind",
        {"nested": {"a": 1}},
        idempotency_key="request",
    )

    with pytest.raises(EventIdempotencyConflictError):
        log.append(
            "stream",
            "kind",
            {"nested": [["a", 1]]},
            idempotency_key="request",
        )

    assert original.payload != freeze_metadata({"nested": [["a", 1]]})
    assert freeze_metadata({"value": True}) != freeze_metadata({"value": 1})
    assert freeze_metadata({"value": 1.0}) != freeze_metadata({"value": 1})
    assert len(log.events()) == 1


def test_explicit_occurred_at_is_part_of_idempotency_identity() -> None:
    log = EventLog(clock=LogicalClock(50))
    original = log.append(
        "stream",
        "kind",
        idempotency_key="request",
        occurred_at=1,
    )

    assert (
        log.append(
            "stream",
            "kind",
            idempotency_key="request",
            occurred_at=1,
        )
        is original
    )
    with pytest.raises(EventIdempotencyConflictError):
        log.append(
            "stream",
            "kind",
            idempotency_key="request",
            occurred_at=999,
        )
    assert len(log.events()) == 1


def test_concurrent_expected_version_has_one_winner() -> None:
    log = EventLog(clock=LogicalClock())

    def append(index: int) -> str:
        try:
            log.append("stream", f"event-{index}", expected_version=0)
        except ExpectedVersionError:
            return "lost"
        return "won"

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(append, range(40)))

    assert results.count("won") == 1
    assert log.version("stream") == 1
    assert log.snapshot().next_sequence == 2


def test_input_validation_covers_payload_and_identity_boundaries() -> None:
    with pytest.raises(TypeError, match="clock"):
        EventLog(clock=object())  # type: ignore[arg-type]
    log = EventLog(clock=LogicalClock())
    before = log.snapshot()
    bad_mapping: Any = {1: "bad"}
    with pytest.raises(ValueError):
        log.append("", "kind")
    with pytest.raises(ValueError):
        log.append("stream", "")
    with pytest.raises(TypeError):
        log.append("stream", "kind", bad_mapping)
    with pytest.raises(ValueError):
        log.append("stream", "kind", {"bad": nan})
    with pytest.raises(ValueError):
        log.append("stream", "kind", expected_version=-1)
    with pytest.raises(TypeError):
        log.append("stream", "kind", expected_version=True)
    with pytest.raises(ValueError):
        log.append("stream", "kind", idempotency_key="")
    with pytest.raises(TypeError):
        log.append("stream", "kind", occurred_at=True)
    assert log.snapshot() == before

    with pytest.raises(TypeError, match="tuple"):
        Event(1, "stream", 1, "kind", 0, [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="key/value"):
        Event(1, "stream", 1, "kind", 0, (("bad",),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="keys"):
        Event(1, "stream", 1, "kind", 0, ((1, "bad"),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        Event(1, "stream", 1, "kind", 0, (("a", 1), ("a", 2)))
    with pytest.raises(ValueError, match="must not be empty"):
        Event(1, "stream", 1, "kind", 0, idempotency_key="")


def test_checkpoint_restores_versions_and_retry_identity() -> None:
    clock = LogicalClock(5)
    log = EventLog(clock=clock)
    original = log.append(
        "stream", "created", {"x": 1}, expected_version=0, idempotency_key="r1"
    )
    log.append("other", "created")
    checkpoint = log.checkpoint()

    restored = EventLog.from_checkpoint(checkpoint, clock=clock)
    assert restored.snapshot() == log.snapshot()
    assert (
        restored.append(
            "stream",
            "created",
            {"x": 1},
            expected_version=0,
            idempotency_key="r1",
        )
        == original
    )
    next_event = restored.append("stream", "next", expected_version=1)
    observed_position = (next_event.sequence, next_event.version)
    expected_position = (3, 2)
    assert observed_position == expected_position

    with pytest.raises(InvalidEventCheckpointError, match="next_sequence"):
        EventLog.from_checkpoint(replace(checkpoint, next_sequence=99), clock=clock)
    with pytest.raises(InvalidEventCheckpointError, match="idempotency requests"):
        EventLog.from_checkpoint(replace(checkpoint, requests=()), clock=clock)
    bad_event = replace(checkpoint.events[0], version=2)
    with pytest.raises(InvalidEventCheckpointError, match="stream versions"):
        EventLog.from_checkpoint(
            replace(checkpoint, events=(bad_event, *checkpoint.events[1:])),
            clock=clock,
        )


def test_checkpoint_rejects_structural_and_retry_corruption() -> None:
    clock = LogicalClock()
    log = EventLog(clock=clock)
    log.append("stream", "one", expected_version=0, idempotency_key="r1")
    log.append("stream", "two", expected_version=1, idempotency_key="r2")
    checkpoint = log.checkpoint()
    request_type = type(checkpoint.requests[0])

    with pytest.raises(TypeError, match="EventLogCheckpoint"):
        EventLog.from_checkpoint(object(), clock=clock)  # type: ignore[arg-type]
    with pytest.raises(InvalidEventCheckpointError, match="positive"):
        EventLog.from_checkpoint(replace(checkpoint, next_sequence=0), clock=clock)
    with pytest.raises(TypeError, match="Event values"):
        EventLog.from_checkpoint(
            replace(checkpoint, events=(object(),)),  # type: ignore[arg-type]
            clock=clock,
        )
    bad_sequence = replace(checkpoint.events[0], sequence=2)
    with pytest.raises(InvalidEventCheckpointError, match="global sequences"):
        EventLog.from_checkpoint(
            replace(checkpoint, events=(bad_sequence, checkpoint.events[1])),
            clock=clock,
        )
    duplicate_key_event = replace(checkpoint.events[1], idempotency_key="r1")
    with pytest.raises(InvalidEventCheckpointError, match="duplicate event"):
        EventLog.from_checkpoint(
            replace(checkpoint, events=(checkpoint.events[0], duplicate_key_event)),
            clock=clock,
        )
    with pytest.raises(TypeError, match="requests"):
        EventLog.from_checkpoint(
            replace(checkpoint, requests=(object(),)),  # type: ignore[arg-type]
            clock=clock,
        )
    with pytest.raises(InvalidEventCheckpointError, match="duplicate event request"):
        EventLog.from_checkpoint(
            replace(
                checkpoint,
                requests=(checkpoint.requests[0], checkpoint.requests[0]),
            ),
            clock=clock,
        )
    outside = replace(checkpoint.requests[0], event_sequence=99)
    with pytest.raises(InvalidEventCheckpointError, match="outside"):
        EventLog.from_checkpoint(
            replace(checkpoint, requests=(outside, checkpoint.requests[1])),
            clock=clock,
        )
    mismatched = replace(checkpoint.requests[0], kind="wrong")
    with pytest.raises(InvalidEventCheckpointError, match="does not match"):
        EventLog.from_checkpoint(
            replace(checkpoint, requests=(mismatched, checkpoint.requests[1])),
            clock=clock,
        )
    wrong_expected = request_type(
        "stream",
        "r1",
        1,
        "one",
        (),
        1,
    )
    with pytest.raises(InvalidEventCheckpointError, match="expected_version"):
        EventLog.from_checkpoint(
            replace(checkpoint, requests=(wrong_expected, checkpoint.requests[1])),
            clock=clock,
        )


def test_direct_event_validation_rejects_invalid_coordinates() -> None:
    with pytest.raises(TypeError, match="integer"):
        Event(True, "stream", 1, "kind", 0)
    with pytest.raises(ValueError, match="positive"):
        Event(1, "stream", 0, "kind", 0)

    mutable = [1]
    event = Event(1, "stream", 1, "kind", 0, (("values", mutable),))
    mutable.append(2)
    expected_payload = freeze_metadata({"values": [1]})
    assert event.payload == expected_payload
