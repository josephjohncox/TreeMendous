"""Contracts for immutable named capacity arithmetic."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from treemendous.applications._shared.capacity import CapacityVector


def test_capacity_vector_is_canonical_immutable_and_detached() -> None:
    source = {"memory": 16, "cpu": 4}
    capacity = CapacityVector(source)
    source["cpu"] = 99

    expected_items = tuple(sorted({"memory": 16, "cpu": 4}.items()))
    expected_dimensions = tuple(name for name, _ in expected_items)
    assert capacity.items_tuple == expected_items
    assert capacity.dimensions == expected_dimensions
    assert capacity["cpu"] == 4
    assert dict(capacity) == {"cpu": 4, "memory": 16}
    assert hash(capacity) == hash(CapacityVector(cpu=4, memory=16))
    with pytest.raises(FrozenInstanceError):
        capacity._items = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        ({}, ValueError, "at least one"),
        ({"": 1}, ValueError, "non-empty"),
        ({"cpu": -1}, ValueError, "non-negative"),
        ({"cpu": True}, TypeError, "integer"),
        ({"cpu": 1.5}, TypeError, "integer"),
    ],
)
def test_capacity_vector_validates_dimensions_and_values(
    values: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        CapacityVector(values)  # type: ignore[arg-type]


def test_capacity_vector_rejects_duplicate_iterable_dimensions() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        CapacityVector((("cpu", 1), ("cpu", 2)))


def test_capacity_arithmetic_and_fits_are_componentwise() -> None:
    available = CapacityVector(cpu=8, memory=32)
    used = CapacityVector(cpu=3, memory=10)

    assert available.fits(used)
    assert used.fits_within(available)
    assert available.subtract(used) == CapacityVector(cpu=5, memory=22)
    assert used.add(CapacityVector(cpu=2, memory=6)) == CapacityVector(cpu=5, memory=16)
    assert available - used == CapacityVector(cpu=5, memory=22)
    assert used + used == CapacityVector(cpu=6, memory=20)

    with pytest.raises(ValueError, match="negative"):
        used.subtract(available)


def test_capacity_operations_require_exact_key_sets() -> None:
    cpu_only = CapacityVector(cpu=2)
    cpu_and_memory = CapacityVector(cpu=2, memory=0)

    for operation in (
        lambda: cpu_only.add(cpu_and_memory),
        lambda: cpu_only.subtract(cpu_and_memory),
        lambda: cpu_only.fits(cpu_and_memory),
    ):
        with pytest.raises(ValueError, match="exactly the same"):
            operation()

    with pytest.raises(TypeError, match="CapacityVector"):
        cpu_only.add({"cpu": 1})  # type: ignore[arg-type]
