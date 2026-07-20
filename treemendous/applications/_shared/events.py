"""Private deterministic in-memory event streams.

The log supplies optimistic per-stream versions and retry idempotency inside one
process.  It is not durable storage, replication, consensus, or a distributed
transaction log; applications must persist/replicate checkpoints externally.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from threading import RLock
from typing import Any

from treemendous.applications._shared.clock import Clock
from treemendous.domain import validate_coordinate


class EventLogError(RuntimeError):
    """Base error raised by the private event log."""


class ExpectedVersionError(EventLogError):
    """Raised when optimistic stream-version validation fails."""


class EventIdempotencyConflictError(EventLogError):
    """Raised when an idempotency key is reused for a different append."""


class InvalidEventCheckpointError(EventLogError):
    """Raised when checkpoint structure or derived versions are inconsistent."""


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


@dataclass(frozen=True)
class _FrozenBoolean:
    """Tagged boolean that cannot compare equal to integer metadata."""

    value: bool


@dataclass(frozen=True)
class _FrozenFloat:
    """Tagged float that cannot compare equal to integer metadata."""

    value: float


@dataclass(frozen=True)
class _FrozenMapping:
    """Tagged immutable representation that cannot alias a sequence."""

    items: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class _FrozenSequence:
    """Tagged immutable representation that cannot alias a mapping."""

    items: tuple[Any, ...]


def _freeze_value(value: Any) -> Any:
    if isinstance(
        value,
        (_FrozenBoolean, _FrozenFloat, _FrozenMapping, _FrozenSequence),
    ):
        return value
    if (
        value is None
        or isinstance(value, (int, str, bytes))
        and not isinstance(value, bool)
    ):
        return value
    if isinstance(value, bool):
        return _FrozenBoolean(value)
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("event payload floats must be finite")
        return _FrozenFloat(value)
    if isinstance(value, Mapping):
        frozen: list[tuple[str, Any]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("event payload mapping keys must be strings")
            frozen.append((key, _freeze_value(item)))
        return _FrozenMapping(tuple(sorted(frozen)))
    if isinstance(value, (list, tuple)):
        return _FrozenSequence(tuple(_freeze_value(item) for item in value))
    raise TypeError(
        "event payload values must be immutable primitives, mappings, or sequences"
    )


def freeze_metadata(
    value: Mapping[str, Any] | None,
) -> tuple[tuple[str, Any], ...]:
    """Return deterministic recursively immutable application metadata."""
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise TypeError("event metadata must be a mapping")
    frozen = _freeze_value(value)
    if not isinstance(frozen, _FrozenMapping):
        raise RuntimeError("event metadata freezing produced an invalid value")
    return frozen.items


@dataclass(frozen=True)
class Event:
    """One immutable append with global and per-stream ordering evidence."""

    sequence: int
    stream: str
    version: int
    kind: str
    occurred_at: int
    payload: tuple[tuple[str, Any], ...] = ()
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        validate_coordinate(self.sequence, "sequence")
        validate_coordinate(self.version, "version")
        validate_coordinate(self.occurred_at, "occurred_at")
        if self.sequence <= 0 or self.version <= 0:
            raise ValueError("event sequence and version must be positive")
        _nonempty(self.stream, "stream")
        _nonempty(self.kind, "kind")
        if self.idempotency_key is not None:
            _nonempty(self.idempotency_key, "idempotency_key")
        if not isinstance(self.payload, tuple):
            raise TypeError("event payload must be a tuple of key/value pairs")
        payload_mapping: dict[str, Any] = {}
        for item in self.payload:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("event payload must contain key/value pairs")
            key, value = item
            if not isinstance(key, str):
                raise TypeError("event payload mapping keys must be strings")
            if key in payload_mapping:
                raise ValueError("event payload keys must be unique")
            payload_mapping[key] = value
        object.__setattr__(self, "payload", freeze_metadata(payload_mapping))


@dataclass(frozen=True)
class _EventRequest:
    stream: str
    key: str
    expected_version: int | None
    kind: str
    payload: tuple[tuple[str, Any], ...]
    event_sequence: int
    occurred_at: int | None = None

    def __post_init__(self) -> None:
        _nonempty(self.stream, "stream")
        _nonempty(self.key, "idempotency_key")
        _nonempty(self.kind, "kind")
        if self.expected_version is not None:
            validate_coordinate(self.expected_version, "expected_version")
            if self.expected_version < 0:
                raise ValueError("expected_version must be non-negative")
        validate_coordinate(self.event_sequence, "event_sequence")
        if self.event_sequence <= 0:
            raise ValueError("event_sequence must be positive")
        if self.occurred_at is not None:
            validate_coordinate(self.occurred_at, "occurred_at")
        if not isinstance(self.payload, tuple):
            raise TypeError("event request payload must contain key/value pairs")
        payload_mapping: dict[str, Any] = {}
        for key, value in self.payload:
            if key in payload_mapping:
                raise ValueError("event request payload keys must be unique")
            payload_mapping[key] = value
        object.__setattr__(self, "payload", freeze_metadata(payload_mapping))


@dataclass(frozen=True)
class EventLogSnapshot:
    """Immutable point-in-time log observation."""

    events: tuple[Event, ...]
    stream_versions: tuple[tuple[str, int], ...]
    next_sequence: int


@dataclass(frozen=True)
class EventLogCheckpoint:
    """Complete event and retry state required for deterministic restoration."""

    events: tuple[Event, ...]
    requests: tuple[_EventRequest, ...]
    next_sequence: int


class EventLog:
    """Thread-safe process-local append-only event streams."""

    def __init__(self, *, clock: Clock) -> None:
        if not hasattr(clock, "now") or not callable(clock.now):
            raise TypeError("clock must provide a callable now()")
        self._clock = clock
        self._events: list[Event] = []
        self._versions: dict[str, int] = {}
        self._requests: dict[tuple[str, str], _EventRequest] = {}
        self._next_sequence = 1
        self._lock = RLock()

    @staticmethod
    def _timestamp(clock: Clock, occurred_at: int | None) -> int:
        value = clock.now() if occurred_at is None else occurred_at
        return validate_coordinate(value, "occurred_at")

    def append(
        self,
        stream: str,
        kind: str,
        payload: Mapping[str, Any] | None = None,
        *,
        expected_version: int | None = None,
        idempotency_key: str | None = None,
        occurred_at: int | None = None,
    ) -> Event:
        """Append atomically or return the event from an identical retry."""
        stream = _nonempty(stream, "stream")
        kind = _nonempty(kind, "kind")
        frozen_payload = freeze_metadata(payload)
        if expected_version is not None:
            validate_coordinate(expected_version, "expected_version")
            if expected_version < 0:
                raise ValueError("expected_version must be non-negative")
        if idempotency_key is not None:
            idempotency_key = _nonempty(idempotency_key, "idempotency_key")
        if occurred_at is not None:
            validate_coordinate(occurred_at, "occurred_at")
        with self._lock:
            request_key = None if idempotency_key is None else (stream, idempotency_key)
            if request_key is not None:
                prior = self._requests.get(request_key)
                if prior is not None:
                    signature = (
                        expected_version,
                        kind,
                        frozen_payload,
                        occurred_at,
                    )
                    if signature != (
                        prior.expected_version,
                        prior.kind,
                        prior.payload,
                        prior.occurred_at,
                    ):
                        raise EventIdempotencyConflictError(
                            "idempotency key was reused for a different append"
                        )
                    return self._events[prior.event_sequence - 1]

            current_version = self._versions.get(stream, 0)
            if expected_version is not None and expected_version != current_version:
                raise ExpectedVersionError(
                    f"stream {stream!r} is at version {current_version}, "
                    f"expected {expected_version}"
                )
            timestamp = self._timestamp(self._clock, occurred_at)
            event = Event(
                sequence=self._next_sequence,
                stream=stream,
                version=current_version + 1,
                kind=kind,
                occurred_at=timestamp,
                payload=frozen_payload,
                idempotency_key=idempotency_key,
            )
            if request_key is not None:
                if idempotency_key is None:
                    raise RuntimeError("idempotency request key is inconsistent")
                self._requests[request_key] = _EventRequest(
                    stream,
                    idempotency_key,
                    expected_version,
                    kind,
                    frozen_payload,
                    event.sequence,
                    occurred_at,
                )
            self._events.append(event)
            self._versions[stream] = event.version
            self._next_sequence += 1
            return event

    def events(self, stream: str | None = None) -> tuple[Event, ...]:
        """Return global order or one stream's version order."""
        with self._lock:
            if stream is None:
                return tuple(self._events)
            stream = _nonempty(stream, "stream")
            return tuple(event for event in self._events if event.stream == stream)

    def version(self, stream: str) -> int:
        """Return zero for an unseen stream."""
        stream = _nonempty(stream, "stream")
        with self._lock:
            return self._versions.get(stream, 0)

    def snapshot(self) -> EventLogSnapshot:
        """Capture deterministic immutable state."""
        with self._lock:
            return EventLogSnapshot(
                tuple(self._events),
                tuple(sorted(self._versions.items())),
                self._next_sequence,
            )

    def checkpoint(self) -> EventLogCheckpoint:
        """Capture all append and retry state."""
        with self._lock:
            return EventLogCheckpoint(
                tuple(self._events),
                tuple(self._requests[key] for key in sorted(self._requests)),
                self._next_sequence,
            )

    @classmethod
    def from_checkpoint(
        cls, checkpoint: EventLogCheckpoint, *, clock: Clock
    ) -> EventLog:
        """Validate and restore without accepting sequence/version forks."""
        if not isinstance(checkpoint, EventLogCheckpoint):
            raise TypeError("checkpoint must be an EventLogCheckpoint")
        validate_coordinate(checkpoint.next_sequence, "next_sequence")
        if checkpoint.next_sequence <= 0:
            raise InvalidEventCheckpointError("next_sequence must be positive")
        candidate = cls(clock=clock)
        versions: dict[str, int] = {}
        for expected_sequence, event in enumerate(checkpoint.events, start=1):
            if not isinstance(event, Event):
                raise TypeError("checkpoint events must be Event values")
            if event.sequence != expected_sequence:
                raise InvalidEventCheckpointError(
                    "event global sequences must be contiguous"
                )
            expected_version = versions.get(event.stream, 0) + 1
            if event.version != expected_version:
                raise InvalidEventCheckpointError(
                    "event stream versions must be contiguous"
                )
            versions[event.stream] = event.version
        if checkpoint.next_sequence != len(checkpoint.events) + 1:
            raise InvalidEventCheckpointError("next_sequence is inconsistent")
        requests: dict[tuple[str, str], _EventRequest] = {}
        events_with_requests = tuple(
            event for event in checkpoint.events if event.idempotency_key is not None
        )
        event_request_keys = {
            (event.stream, event.idempotency_key): event.sequence
            for event in events_with_requests
        }
        if len(event_request_keys) != len(events_with_requests):
            raise InvalidEventCheckpointError(
                "duplicate event idempotency key in checkpoint"
            )
        for request in checkpoint.requests:
            if not isinstance(request, _EventRequest):
                raise TypeError("checkpoint requests are invalid")
            key = (request.stream, request.key)
            if key in requests:
                raise InvalidEventCheckpointError("duplicate event request key")
            if request.expected_version is not None:
                validate_coordinate(request.expected_version, "expected_version")
                if request.expected_version < 0:
                    raise InvalidEventCheckpointError(
                        "event expected_version must be non-negative"
                    )
            if not (1 <= request.event_sequence < checkpoint.next_sequence):
                raise InvalidEventCheckpointError("event request points outside log")
            event = checkpoint.events[request.event_sequence - 1]
            if (
                event.stream != request.stream
                or event.kind != request.kind
                or event.payload != request.payload
                or event.idempotency_key != request.key
                or (
                    request.occurred_at is not None
                    and event.occurred_at != request.occurred_at
                )
            ):
                raise InvalidEventCheckpointError(
                    "event request does not match its event"
                )
            if (
                request.expected_version is not None
                and request.expected_version != event.version - 1
            ):
                raise InvalidEventCheckpointError(
                    "event request expected_version does not match its event"
                )
            requests[key] = request
        if {
            key: request.event_sequence for key, request in requests.items()
        } != event_request_keys:
            raise InvalidEventCheckpointError(
                "event idempotency requests do not completely match the log"
            )

        candidate._events = list(checkpoint.events)
        candidate._versions = versions
        candidate._requests = requests
        candidate._next_sequence = checkpoint.next_sequence
        return candidate
