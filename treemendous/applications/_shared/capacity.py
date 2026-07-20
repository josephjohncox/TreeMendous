"""Immutable capacity arithmetic for private application engines."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, init=False)
class CapacityVector(Mapping[str, int]):
    """A deterministic vector of named, non-negative integer capacities.

    Dimension names are part of the type of a vector: arithmetic and ``fits``
    require *exactly* the same key set.  This deliberately rejects accidental
    comparisons such as a CPU-only demand against a CPU-and-memory capacity.
    Values are stored as a sorted tuple, so instances are immutable, hashable,
    and deterministic regardless of input mapping order.
    """

    _items: tuple[tuple[str, int], ...]

    def __init__(
        self,
        values: Mapping[str, int] | Iterable[tuple[str, int]] | None = None,
        **dimensions: int,
    ) -> None:
        if values is not None and dimensions:
            raise TypeError("provide either values or named dimensions, not both")
        source: Iterable[tuple[str, int]]
        if values is None:
            source = dimensions.items()
        elif isinstance(values, Mapping):
            source = values.items()
        else:
            source = values

        normalized: dict[str, int] = {}
        for name, value in source:
            if not isinstance(name, str) or not name:
                raise ValueError("capacity dimension names must be non-empty strings")
            if name in normalized:
                raise ValueError(f"duplicate capacity dimension: {name!r}")
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"capacity {name!r} must be an integer")
            if value < 0:
                raise ValueError(f"capacity {name!r} must be non-negative")
            normalized[name] = value
        if not normalized:
            raise ValueError("a capacity vector must have at least one dimension")
        object.__setattr__(self, "_items", tuple(sorted(normalized.items())))

    def __getitem__(self, name: str) -> int:
        for candidate, value in self._items:
            if candidate == name:
                return value
        raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def dimensions(self) -> tuple[str, ...]:
        """Return dimension names in canonical order."""
        return tuple(self)

    @property
    def items_tuple(self) -> tuple[tuple[str, int], ...]:
        """Return the immutable canonical representation."""
        return self._items

    def to_dict(self) -> dict[str, int]:
        """Return a detached mutable representation."""
        return dict(self._items)

    def add(self, other: CapacityVector) -> CapacityVector:
        """Add vectors after enforcing identical dimensions."""
        self._require_same_dimensions(other)
        return CapacityVector((name, value + other[name]) for name, value in self._items)

    def subtract(self, other: CapacityVector) -> CapacityVector:
        """Subtract ``other``, rejecting a result with a negative component."""
        self._require_same_dimensions(other)
        if not self.fits(other):
            raise ValueError("capacity subtraction would produce a negative value")
        return CapacityVector((name, value - other[name]) for name, value in self._items)

    def fits(self, required: CapacityVector) -> bool:
        """Return whether this vector can satisfy ``required`` in every dimension."""
        self._require_same_dimensions(required)
        return all(value >= required[name] for name, value in self._items)

    def fits_within(self, available: CapacityVector) -> bool:
        """Return whether this vector can be satisfied by ``available``."""
        return available.fits(self)

    def __add__(self, other: CapacityVector) -> CapacityVector:
        return self.add(other)

    def __sub__(self, other: CapacityVector) -> CapacityVector:
        return self.subtract(other)

    def _require_same_dimensions(self, other: CapacityVector) -> None:
        if not isinstance(other, CapacityVector):
            raise TypeError("capacity operations require another CapacityVector")
        if self.dimensions != other.dimensions:
            raise ValueError(
                "capacity vectors must have exactly the same dimension keys"
            )
