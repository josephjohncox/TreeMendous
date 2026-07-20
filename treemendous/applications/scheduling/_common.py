"""Shared validation and deterministic placement values for scheduling engines.

The scenario engines are intentionally in-memory policy components.  They make
one-process transitions atomic through :class:`ReservationLedger`; they are not
optimizers, durable stores, distributed locks, or consensus systems.
"""

from __future__ import annotations

from dataclasses import dataclass

from treemendous.applications._shared.reservations import (
    Reservation,
    ReservationConflict,
)
from treemendous.domain import Span, validate_coordinate, validate_length


def text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def integer(value: int, name: str, *, minimum: int = 0) -> int:
    validate_coordinate(value, name)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def positive(value: int, name: str) -> int:
    validate_length(value)
    return value


def names(values: frozenset[str], name: str) -> frozenset[str]:
    if not isinstance(values, frozenset):
        raise TypeError(f"{name} must be a frozenset")
    for value in values:
        text(value, name)
    return values


def spans(values: tuple[Span, ...], name: str) -> tuple[Span, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a nonempty tuple of Span values")
    if not all(isinstance(value, Span) for value in values):
        raise TypeError(f"{name} must contain Span values")
    return tuple(sorted(values))


@dataclass(frozen=True)
class Placement:
    """A deterministic selected resource and its committed reservation."""

    resource: str
    reservation: Reservation

    @property
    def start(self) -> int:
        return self.reservation.start

    @property
    def end(self) -> int:
        return self.reservation.end

    @property
    def id(self) -> str:
        return self.reservation.id


class SchedulingUnavailableError(ValueError):
    """No compatible resource could satisfy a bounded scheduling request."""

    def __init__(
        self,
        message: str,
        *,
        conflicts: tuple[ReservationConflict, ...] = (),
        considered: tuple[str, ...] = (),
    ) -> None:
        self.conflicts = conflicts
        self.considered = considered
        super().__init__(message)
