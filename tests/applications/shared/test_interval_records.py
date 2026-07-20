"""Contracts for the private identity-preserving interval record index."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event, Thread
from typing import Any
from uuid import uuid4

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

    assert first.owner == "owner-a"
    assert first.sequence == 1
    assert second.owner == "owner-a"
    assert second.sequence == 2
    assert third.owner == "owner-b"
    assert third.sequence == 1
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

    index.remove(second, owner="owner-a")
    fourth = index.insert("owner-a", 10, 20, {"value": 4})
    assert fourth.owner == "owner-a"
    assert fourth.sequence == 3
    assert [record.handle for record in index.overlaps(10, 20)] == [
        first,
        third,
        fourth,
    ]


def test_handles_are_index_scoped_value_identities_not_authorization() -> None:
    first_index = IntervalRecordIndex[str, str](str)
    second_index = IntervalRecordIndex[str, str](str)
    local = first_index.insert("owner", 0, 1, "local")
    foreign = second_index.insert("owner", 0, 1, "foreign")
    reconstructed = RecordHandle(local.owner, local.sequence, local.lineage)
    forged = RecordHandle(local.owner, local.sequence, uuid4())

    assert local != foreign
    assert reconstructed == local
    assert first_index.get(reconstructed).payload == "local"
    for invalid in (foreign, forged):
        with pytest.raises(KeyError):
            first_index.get(invalid)
        with pytest.raises(KeyError):
            first_index.update(invalid, owner="owner", payload="changed")
        with pytest.raises(KeyError):
            first_index.remove(invalid, owner="owner")

    with pytest.raises(PermissionError, match="owner"):
        first_index.update(local, owner="intruder", payload="changed")
    with pytest.raises(PermissionError, match="owner"):
        first_index.remove(local, owner="intruder")
    assert first_index.remove(reconstructed, owner="owner").payload == "local"
    with pytest.raises(KeyError):
        first_index.update(local, owner="owner", payload="changed")
    with pytest.raises(KeyError):
        first_index.remove(local, owner="owner")


def test_update_preserves_identity_and_original_order() -> None:
    index = IntervalRecordIndex[str, list[int]](deepcopy)
    first = index.insert("a", 0, 2, [1])
    second = index.insert("b", 5, 7, [2])

    updated = index.update(first, owner="a", start=5, end=8, payload=[3])

    assert updated.handle == first
    assert updated.insertion_order == 0
    assert updated.payload == [3]
    assert [record.handle for record in index.overlaps(6, 7)] == [first, second]
    assert index.get(first).span.start == 5

    with pytest.raises(ValueError, match="start < end"):
        index.update(first, owner="a", start=9)
    with pytest.raises(ValueError, match="requires"):
        index.update(first, owner="a")
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
        index.update(first, owner="owner", payload={"value": 3})
    with pytest.raises(RuntimeError, match="clone failed"):
        index.remove(first, owner="owner")
    fail = False

    assert index.snapshot() == before
    inserted = index.insert("owner", 2, 4, {"value": 2})
    assert inserted.owner == "owner"
    assert inserted.sequence == 2


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
        index.update(handle, owner="owner", payload={"value": 2})

    fail_at = None
    assert index.get(handle).payload == {"value": 1}


def test_reentrant_cloner_mutation_linearizes_before_pending_update() -> None:
    armed = False
    index: IntervalRecordIndex[str, dict[str, int]]

    def cloner(payload: dict[str, int]) -> dict[str, int]:
        nonlocal armed
        if armed:
            armed = False
            index.insert("nested", 10, 11, {"value": 99})
        return deepcopy(payload)

    index = IntervalRecordIndex(cloner)
    handle = index.insert("owner", 0, 2, {"value": 1})

    armed = True
    updated = index.update(handle, owner="owner", payload={"value": 2})

    assert updated.payload == {"value": 2}
    assert [record.handle.owner for record in index.snapshot().records] == [
        "owner",
        "nested",
    ]


def test_update_retries_only_when_its_record_changes_during_cloning() -> None:
    clone_started = Event()
    clone_may_finish = Event()
    block_outer_once = True
    results: list[Any] = []
    errors: list[BaseException] = []

    def cloner(payload: str) -> str:
        nonlocal block_outer_once
        if payload == "outer" and block_outer_once:
            block_outer_once = False
            clone_started.set()
            if not clone_may_finish.wait(timeout=2):
                raise RuntimeError("test did not release payload clone")
        return payload

    index = IntervalRecordIndex[str, str](cloner)
    handle = index.insert("owner", 0, 1, "initial")

    def update_outer() -> None:
        try:
            results.append(index.update(handle, owner="owner", payload="outer"))
        except BaseException as exc:  # capture worker evidence
            errors.append(exc)

    worker = Thread(target=update_outer)
    worker.start()
    assert clone_started.wait(timeout=1)
    index.update(handle, owner="owner", start=5, end=6, payload="inner")
    clone_may_finish.set()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert not errors
    assert results[0].span.start == 5
    assert results[0].span.end == 6
    assert results[0].payload == "outer"
    assert index.get(handle) == results[0]


def test_cross_thread_mutation_from_cloner_completes_without_deadlock() -> None:
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

    assert handle.owner == "owner"
    assert handle.sequence == 1
    assert not errors
    assert len(index) == 2


def test_cross_thread_read_from_cloner_completes_without_deadlock() -> None:
    armed = False
    observations: list[int] = []
    index: IntervalRecordIndex[str, str]

    def cloner(payload: str) -> str:
        nonlocal armed
        if armed:
            armed = False
            worker = Thread(
                target=lambda: observations.append(index.diagnostics().record_count)
            )
            worker.start()
            worker.join(timeout=1)
            if worker.is_alive():
                raise RuntimeError("cross-thread payload read deadlocked")
        return payload

    index = IntervalRecordIndex(cloner)
    handle = index.insert("owner", 0, 1, "value")

    armed = True
    assert index.get(handle).payload == "value"
    assert observations == [1]


def test_snapshot_clones_outside_lock_while_unrelated_writer_commits() -> None:
    armed = False
    clone_started = Event()
    clone_may_finish = Event()
    snapshots: list[Any] = []
    errors: list[BaseException] = []

    def cloner(payload: str) -> str:
        if armed and payload == "blocked":
            clone_started.set()
            if not clone_may_finish.wait(timeout=2):
                raise RuntimeError("test did not release payload clone")
        return payload

    index = IntervalRecordIndex[str, str](cloner)
    original = index.insert("reader", 0, 1, "blocked")
    armed = True

    def read_snapshot() -> None:
        try:
            snapshots.append(index.snapshot())
        except BaseException as exc:  # capture worker evidence
            errors.append(exc)

    reader = Thread(target=read_snapshot)
    reader.start()
    assert clone_started.wait(timeout=1)
    try:
        inserted = index.insert("writer", 1, 2, "safe")
    finally:
        clone_may_finish.set()
        reader.join(timeout=1)

    assert not reader.is_alive()
    assert not errors
    assert [record.handle for record in snapshots[0].records] == [original]
    assert [record.handle for record in index.snapshot().records] == [
        original,
        inserted,
    ]


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
    missing = RecordHandle("owner", 1, uuid4())
    with pytest.raises(KeyError):
        index.get(missing)
    with pytest.raises(KeyError):
        index.remove(missing, owner="owner")
    assert index.snapshot() == before
