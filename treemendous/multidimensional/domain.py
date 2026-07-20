"""Experimental identity-preserving multidimensional domain values."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from math import prod
from typing import Any
from uuid import UUID

from treemendous.domain import validate_coordinate


@dataclass(frozen=True, init=False)
class Box:
    """A nonempty axis-aligned half-open integer box."""

    lower: tuple[int, ...]
    upper: tuple[int, ...]

    def __init__(self, lower: Iterable[int], upper: Iterable[int]) -> None:
        normalized_lower = tuple(lower)
        normalized_upper = tuple(upper)
        if not normalized_lower:
            raise ValueError("box must contain at least one dimension")
        if len(normalized_lower) != len(normalized_upper):
            raise ValueError("box lower and upper bounds must have equal dimensions")
        for axis, (lower_bound, upper_bound) in enumerate(
            zip(normalized_lower, normalized_upper, strict=True)
        ):
            validate_coordinate(lower_bound, f"lower[{axis}]")
            validate_coordinate(upper_bound, f"upper[{axis}]")
            if lower_bound >= upper_bound:
                raise ValueError(f"box axis {axis} must satisfy lower < upper")
        object.__setattr__(self, "lower", normalized_lower)
        object.__setattr__(self, "upper", normalized_upper)

    @property
    def dimensions(self) -> int:
        return len(self.lower)

    @property
    def volume(self) -> int:
        return prod(
            upper_bound - lower_bound
            for lower_bound, upper_bound in zip(self.lower, self.upper, strict=True)
        )

    def _require_dimensions(self, other: Box) -> None:
        if self.dimensions != other.dimensions:
            raise ValueError("boxes must have matching dimensions")

    def contains(self, other: Box) -> bool:
        """Return whether this box contains all points of ``other``."""
        self._require_dimensions(other)
        return all(
            left_lower <= right_lower and right_upper <= left_upper
            for left_lower, left_upper, right_lower, right_upper in zip(
                self.lower,
                self.upper,
                other.lower,
                other.upper,
                strict=True,
            )
        )

    def overlaps(self, other: Box) -> bool:
        """Return whether the half-open boxes share at least one point."""
        self._require_dimensions(other)
        return all(
            left_lower < right_upper and right_lower < left_upper
            for left_lower, left_upper, right_lower, right_upper in zip(
                self.lower,
                self.upper,
                other.lower,
                other.upper,
                strict=True,
            )
        )


@dataclass(frozen=True)
class BoxHandle:
    """Owner-scoped value identity; it is not an authorization capability."""

    sequence: int
    _owner: UUID = field(repr=False)


@dataclass(frozen=True)
class BoxEntry:
    """One identity-preserving indexed box and its application payload."""

    handle: BoxHandle
    box: Box
    data: Any = None


def _detached_entry(
    entry: BoxEntry,
    cloner: Callable[[Any], Any] = deepcopy,
) -> BoxEntry:
    return BoxEntry(entry.handle, entry.box, cloner(entry.data))


@dataclass(frozen=True)
class BoxIndexSnapshot:
    """Detached point-in-time membership and payload snapshot."""

    dimensions: int
    version: int
    entries: tuple[BoxEntry, ...]
    _payload_cloner: Callable[[Any], Any] = field(
        default=deepcopy,
        repr=False,
        compare=False,
    )

    def __deepcopy__(self, memo: dict[int, Any]) -> BoxIndexSnapshot:
        existing = memo.get(id(self))
        if existing is not None:
            return existing
        copied = BoxIndexSnapshot(
            self.dimensions,
            self.version,
            tuple(
                _detached_entry(entry, self._payload_cloner) for entry in self.entries
            ),
            self._payload_cloner,
        )
        memo[id(self)] = copied
        return copied

    def _validate_box(self, box: Box) -> None:
        if box.dimensions != self.dimensions:
            raise ValueError("box dimensions must match the snapshot")

    def get(self, handle: BoxHandle) -> BoxEntry:
        for entry in self.entries:
            if entry.handle == handle:
                return _detached_entry(entry, self._payload_cloner)
        raise KeyError(handle)

    def overlaps(self, box: Box) -> tuple[BoxEntry, ...]:
        self._validate_box(box)
        return tuple(
            _detached_entry(entry, self._payload_cloner)
            for entry in self.entries
            if entry.box.overlaps(box)
        )
