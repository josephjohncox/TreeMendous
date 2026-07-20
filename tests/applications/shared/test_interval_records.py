"""Contracts for the private identity-preserving interval record index."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Thread
from typing import Any

import pytest

from treemendous.applications._shared.interval_records import (
    IntervalRecordIndex,
    RecordHandle,
)


def test_coincident_records_keep_identity_and_insertion_order() -> None:
    index = IntervalRecordIndex[str, dict[str, int]](deepcopy)
    first = index.insert("owner-a", 10, 20, {"value": 1})
    second = index.insert("owner-a", 10, 20, {"value": 2})
    third = index.insert("owner-b", 10, 20, {"value": 3})

    assert first == RecordHandle("owner-a", 1)
    assert second == RecordHandle("owner-a", 2)
    assert third == RecordHandle("owner-b", 1)
    assert [record.handle for record in index.overlaps(15, 16)] == [
        first,
        second,
        third,
    ]
    assert [record.handle for record in index.containing(10)] == [
        first,
        second,
        third,
    ]
    assert not index.containing(20)

    index.remove(second)
    fourth = index.insert("owner-a", 10, 20, {"value": 4})
    assert fourth == RecordHandle("owner-a", 3)
    assert [record.handle for record in index.overlaps(10, 20)] == [
        first,
        third,
        fourth,
    ]


def test_update_preserves_identity_and_original_order() -> None:
    index = IntervalRecordIndex[str, list[int]](deepcopy)
    first = index.insert("a", 0, 2, [1])
    second = index.insert("b", 5, 7, [2])

    updated = index.update(first, start=5, end=8, payload=[3])

    assert updated.handle == first
    assert updated.insertion_order == 0
    assert updated.payload == [3]
    assert [record.handle for record in index.overlaps(6, 7)] == [first, second]
    assert index.get(first).span.start == 5

    with pytest.raises(ValueError, match="start < end"):
        index.update(first, start=9)
    with pytest.raises(ValueError, match="requires"):
        index.update(first)
    assert index.get(first) == updated


def test_payloads_are_detached_at_every_boundary() -> None:
    index = IntervalRecordIndex[str, dict[str, list[int]]](deepcopy)
    source = {"items": [1]}
    handle = index.insert("owner", 0, 4, source)
    source["items"].append(2)
    observed = index.get(handle)
    assert observed.payload == {"items": [1]}

    observed.payload["items"].append(3)
    queried = index.overlaps(0, 1)[0]
    queried.payload["items"].append(4)
    snapped = index.snapshot().records[0]
    snapped.payload["items"].append(5)
    assert index.get(handle).payload == {"items": [1]}


def test_cloner_failure_is_atomic() -> None:
    fail = False

    def cloner(payload: dict[str, Any]) -> dict[str, Any]:
        if fail:
            raise RuntimeError("clone failed")
        return deepcopy(payload)

    index = IntervalRecordIndex[str, dict[str, Any]](cloner)
    first = index.insert("owner", 0, 2, {"value": 1})
    before = index.snapshot()

    fail = True
    with pytest.raises(RuntimeError, match="clone failed"):
        index.insert("owner", 2, 4, {"value": 2})
    with pytest.raises(RuntimeError, match="clone failed"):
        index.update(first, payload={"value": 3})
    with pytest.raises(RuntimeError, match="clone failed"):
        index.remove(first)
    fail = False

    assert index.snapshot() == before
    assert index.insert("owner", 2, 4, {"value": 2}) == RecordHandle("owner", 2)


def test_update_second_clone_failure_does_not_commit() -> None:
    calls = 0
    fail_at: int | None = None

    def cloner(payload: dict[str, int]) -> dict[str, int]:
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise RuntimeError("clone failed")
        return deepcopy(payload)

    index = IntervalRecordIndex[str, dict[str, int]](cloner)
    handle = index.insert("owner", 0, 2, {"value": 1})
    fail_at = calls + 2

    with pytest.raises(RuntimeError, match="clone failed"):
        index.update(handle, payload={"value": 2})

    fail_at = None
    assert index.get(handle).payload == {"value": 1}


def test_cloner_reentrant_mutation_is_rejected_atomically() -> None:
    armed = False
    index: IntervalRecordIndex[str, dict[str, int]]

    def cloner(payload: dict[str, int]) -> dict[str, int]:
        if armed:
            index.insert("nested", 10, 11, {"value": 99})
        return deepcopy(payload)

    index = IntervalRecordIndex(cloner)
    handle = index.insert("owner", 0, 2, {"value": 1})
    before = index.snapshot()

    armed = True
    with pytest.raises(RuntimeError, match="during payload cloning"):
        index.update(handle, payload={"value": 2})
    armed = False

    assert index.snapshot() == before
    assert index.insert("nested", 10, 11, {"value": 99}) == RecordHandle("nested", 1)


def test_cross_thread_mutation_from_cloner_is_rejected_without_deadlock() -> None:
    errors: list[BaseException] = []
    index: IntervalRecordIndex[str, str]

    def cloner(payload: str) -> str:
        if payload == "trigger":

            def mutate() -> None:
                try:
                    index.insert("nested", 2, 3, "safe")
                except BaseException as exc:  # capture worker evidence
                    errors.append(exc)

            worker = Thread(target=mutate)
            worker.start()
            worker.join(timeout=1)
            if worker.is_alive():
                raise RuntimeError("cross-thread payload mutation deadlocked")
        return payload

    index = IntervalRecordIndex(cloner)
    handle = index.insert("owner", 0, 1, "trigger")

    assert handle == RecordHandle("owner", 1)
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "payload cloning" in str(errors[0])
    assert len(index) == 1


def test_diagnostics_snapshot_and_concurrent_owner_sequences() -> None:
    index = IntervalRecordIndex[str, int](lambda value: value)
    with ThreadPoolExecutor(max_workers=8) as executor:
        handles = list(
            executor.map(
                lambda value: index.insert("owner", value, value + 1, value), range(100)
            )
        )

    assert sorted(handle.sequence for handle in handles) == list(range(1, 101))
    diagnostics = index.diagnostics()
    assert diagnostics.record_count == 100
    assert diagnostics.owner_count == 1
    assert diagnostics.next_insertion_order == 100
    snapshot = index.snapshot()
    assert len(snapshot.next_sequences) == 1
    next_owner, next_sequence = snapshot.next_sequences[0]
    assert next_owner == "owner"
    assert next_sequence == 101
    assert len(snapshot.records) == 100


def test_validation_and_missing_handles_do_not_mutate_state() -> None:
    index = IntervalRecordIndex[str, object](deepcopy)
    before = index.snapshot()
    with pytest.raises(ValueError, match="start < end"):
        index.insert("owner", 3, 3, object())
    with pytest.raises(TypeError, match="hashable"):
        index.insert([], 0, 1, object())  # type: ignore[arg-type]
    missing = RecordHandle("owner", 1)
    with pytest.raises(KeyError):
        index.get(missing)
    with pytest.raises(KeyError):
        index.remove(missing)
    assert index.snapshot() == before
