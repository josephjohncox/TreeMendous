"""Common identity and geometry contracts for every multidimensional index."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import pytest

import treemendous
from treemendous.multidimensional import (
    BoundedBoxIndex,
    Box,
    BoxHandle,
    BoxIndex,
    BoxIndex2D,
    BoxIndex3D,
    BoxIndex4D,
    BoxIndexProtocol,
)

IndexFactory = Callable[[], BoxIndexProtocol]


@pytest.fixture(
    params=[
        pytest.param(lambda: BoxIndex(2), id="linear"),
        pytest.param(BoxIndex2D, id="projection-2d"),
        pytest.param(BoxIndex3D, id="projection-3d"),
        pytest.param(BoxIndex4D, id="projection-4d"),
        pytest.param(
            lambda: BoundedBoxIndex(Box((-20, -20), (20, 20)), (4, 4)),
            id="bounded-grid",
        ),
    ]
)
def index_factory(request: pytest.FixtureRequest) -> IndexFactory:
    return request.param


def _box(dimensions: int, lower: int, upper: int) -> Box:
    return Box((lower,) * dimensions, (upper,) * dimensions)


def test_common_duplicate_update_remove_snapshot_contract(
    index_factory: IndexFactory,
) -> None:
    index = index_factory()
    duplicate = _box(index.dimensions, 0, 4)
    first = index.insert(duplicate, {"value": [1]})
    second = index.insert(duplicate, {"value": [2]})
    snapshot = index.snapshot()

    assert [entry.handle for entry in index.overlaps(_box(index.dimensions, 1, 2))] == [
        first,
        second,
    ]
    updated = index.update(first, box=_box(index.dimensions, 8, 10), data=None)
    assert updated.handle == first
    assert updated.data is None
    assert [entry.handle for entry in index.entries()] == [first, second]
    assert index.remove(second).handle == second
    third = index.insert(duplicate, "third")

    assert third.sequence == 3
    assert [entry.handle for entry in index.entries()] == [first, third]
    assert [entry.handle for entry in snapshot.entries] == [first, second]
    diagnostics = index.diagnostics()
    assert diagnostics.entry_count == 2
    assert diagnostics.duplicate_entry_count == 0
    assert diagnostics.version == 5


def test_common_half_open_and_huge_integer_contract(
    index_factory: IndexFactory,
) -> None:
    index = index_factory()
    # The bounded implementation has finite configured bounds, while every
    # unbounded implementation must also handle arbitrary-size integers.
    coordinate = 10 if isinstance(index, BoundedBoxIndex) else 10**100
    indexed = _box(index.dimensions, 0, coordinate)
    handle = index.insert(indexed, "indexed")

    assert not index.overlaps(_box(index.dimensions, coordinate, coordinate + 1))
    assert [entry.handle for entry in index.overlaps(_box(index.dimensions, 1, 2))] == [
        handle
    ]


def test_common_owner_scope_and_payload_detachment(
    index_factory: IndexFactory,
) -> None:
    first = index_factory()
    second = index_factory()
    payload = {"nested": [1]}
    local = first.insert(_box(first.dimensions, 0, 2), payload)
    foreign = second.insert(_box(second.dimensions, 0, 2), "foreign")
    payload["nested"].append(2)

    with pytest.raises(KeyError):
        first.get(foreign)
    detached = first.get(local)
    detached.data["nested"].append(3)
    assert first.get(local).data == {"nested": [1]}


def test_optimized_names_are_only_in_multidimensional_namespace() -> None:
    expected = {
        "BoundedBoxIndex",
        "BoxIndex2D",
        "BoxIndex3D",
        "BoxIndex4D",
    }
    assert expected.isdisjoint(treemendous.__all__)
    assert all(not hasattr(treemendous, name) for name in expected)


def test_all_classes_structurally_implement_protocol() -> None:
    indexes: tuple[BoxIndexProtocol, ...] = (
        BoxIndex(2),
        BoxIndex2D(),
        BoxIndex3D(),
        BoxIndex4D(),
        BoundedBoxIndex(Box((0, 0), (4, 4)), (1, 1)),
    )
    assert [index.dimensions for index in indexes] == [2, 2, 3, 4, 2]


def test_fixed_constructor_payload_cloner_is_honored() -> None:
    calls: list[Any] = []

    def cloner(value: Any) -> Any:
        calls.append(value)
        return deepcopy(value)

    index = BoxIndex2D(payload_cloner=cloner)
    handle = index.insert(Box((0, 0), (1, 1)), {"value": 1})
    assert index.get(handle).data == {"value": 1}
    assert len(calls) == 2


def test_handle_annotation_remains_owner_scoped() -> None:
    index = BoxIndex2D()
    handle: BoxHandle = index.insert(Box((0, 0), (1, 1)))
    assert handle.sequence == 1
