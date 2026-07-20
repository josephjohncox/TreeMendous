"""Identity, ordering, snapshot, and diagnostics contracts for BoxIndex."""

from __future__ import annotations

import pytest

import treemendous
from treemendous.multidimensional import (
    Box,
    BoxIndex,
    BoxIndexDiagnostics,
)


def test_duplicate_geometry_preserves_identity_and_insertion_order() -> None:
    index = BoxIndex(2)
    box = Box((0, 0), (10, 10))
    first = index.insert(box, {"name": "first"})
    second = index.insert(box, {"name": "second"})

    assert first != second
    sequences = first.sequence, second.sequence
    expected_sequences = 1, 2
    assert sequences == expected_sequences
    matches = index.overlaps(Box((5, 5), (6, 6)))
    assert [entry.handle for entry in matches] == [first, second]
    assert [entry.data["name"] for entry in matches] == ["first", "second"]
    diagnostics = index.diagnostics()
    expected = BoxIndexDiagnostics("linear", 2, 2, 2, 1, 1)
    assert diagnostics == expected


def test_update_preserves_identity_order_and_accepts_none_payload() -> None:
    index = BoxIndex(2)
    first = index.insert(Box((0, 0), (2, 2)), "first")
    second = index.insert(Box((3, 3), (5, 5)), "second")

    updated = index.update(
        first,
        box=Box((10, 10), (12, 12)),
        data=None,
    )
    assert updated.handle == first
    assert updated.data is None
    assert [entry.handle for entry in index.entries()] == [first, second]
    assert index.diagnostics().version == 3

    with pytest.raises(ValueError, match="requires"):
        index.update(first)


def test_remove_targets_one_duplicate_and_sequences_are_never_reused() -> None:
    index = BoxIndex(2)
    box = Box((0, 0), (2, 2))
    first = index.insert(box, "first")
    second = index.insert(box, "second")

    removed = index.remove(first)
    third = index.insert(box, "third")

    assert removed.handle == first
    assert [entry.handle for entry in index.entries()] == [second, third]
    assert third.sequence == 3
    with pytest.raises(KeyError):
        index.get(first)
    with pytest.raises(KeyError):
        index.remove(first)


def test_foreign_handles_and_wrong_dimensions_are_rejected_atomically() -> None:
    first = BoxIndex(2)
    second = BoxIndex(2)
    local = first.insert(Box((0, 0), (1, 1)), "local")
    foreign = second.insert(Box((0, 0), (1, 1)), "foreign")
    before = first.snapshot()

    for operation in (first.get, first.remove):
        with pytest.raises(KeyError):
            operation(foreign)
    with pytest.raises(KeyError):
        first.update(foreign, data="changed")
    with pytest.raises(ValueError, match="dimensions"):
        first.overlaps(Box((0, 0, 0), (1, 1, 1)))
    with pytest.raises(ValueError, match="dimensions"):
        first.update(local, box=Box((0, 0, 0), (1, 1, 1)))

    assert first.snapshot() == before


def test_snapshot_is_point_in_time_and_supports_detached_queries() -> None:
    index = BoxIndex(2)
    first = index.insert(Box((0, 0), (4, 4)), {"values": [1]})
    snapshot = index.snapshot()
    index.update(first, data={"values": [2]})
    index.insert(Box((10, 10), (12, 12)), "later")

    captured = snapshot.get(first)
    captured.data["values"].append(99)
    assert snapshot.entries[0].data == {"values": [1]}
    matches = snapshot.overlaps(Box((1, 1), (2, 2)))
    assert [entry.handle for entry in matches] == [first]
    assert snapshot.version == 1
    assert len(snapshot.entries) == 1


def test_boxindex_requires_multidimensional_positive_integer_dimension() -> None:
    for invalid in (False, 0, 1, -1):
        with pytest.raises((TypeError, ValueError)):
            BoxIndex(invalid)
    with pytest.raises(TypeError):
        BoxIndex(2.0)  # type: ignore[arg-type]


def test_experimental_surface_does_not_expand_stable_root_api() -> None:
    experimental_names = {
        "Box",
        "BoxEntry",
        "BoxHandle",
        "BoxIndex",
        "BoxIndexDiagnostics",
        "BoxIndexProtocol",
        "BoxIndexSnapshot",
    }
    assert experimental_names.isdisjoint(treemendous.__all__)
    assert all(not hasattr(treemendous, name) for name in experimental_names)

    index = BoxIndex(2)
    for unsupported in ("allocate", "first_fit", "union", "discard", "nearest"):
        assert not hasattr(index, unsupported)
