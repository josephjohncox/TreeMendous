"""Injectable integer clocks for deterministic application engines."""

from __future__ import annotations

from threading import RLock
from typing import Protocol

from treemendous.domain import validate_coordinate


class Clock(Protocol):
    """Minimal clock required by lease and retry state machines."""

    def now(self) -> int: ...


class LogicalClock:
    """Thread-safe monotonic integer clock controlled by tests/applications."""

    def __init__(self, initial: int = 0) -> None:
        self._value = validate_coordinate(initial, "initial")
        self._lock = RLock()

    def now(self) -> int:
        """Return the current logical timestamp."""
        with self._lock:
            return self._value

    def advance(self, delta: int = 1) -> int:
        """Advance by a positive integer and return the new timestamp."""
        validate_coordinate(delta, "delta")
        if delta <= 0:
            raise ValueError("delta must be greater than zero")
        with self._lock:
            self._value += delta
            return self._value

    def set(self, value: int) -> int:
        """Move to ``value`` without allowing time to run backwards."""
        validate_coordinate(value, "value")
        with self._lock:
            if value < self._value:
                raise ValueError("logical time cannot move backwards")
            self._value = value
            return self._value
