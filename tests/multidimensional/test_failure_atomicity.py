"""Payload isolation, copy-failure, reentrancy, and locking tests."""

from __future__ import annotations

from copy import deepcopy
from threading import Event, Thread
from typing import Any

import pytest

from treemendous.multidimensional import Box, BoxIndex


class CopyController:
    def __init__(self) -> None:
        self.calls = 0
        self.fail_at: int | None = None


class ControlledPayload:
    def __init__(self, controller: CopyController, value: str) -> None:
        self.controller = controller
        self.value = value

    def __deepcopy__(self, memo: dict[int, Any]) -> ControlledPayload:
        self.controller.calls += 1
        if self.controller.calls == self.controller.fail_at:
            raise RuntimeError("copy failed")
        return ControlledPayload(self.controller, self.value)


class PausingPayload:
    def __init__(self, entered: Event, proceed: Event, value: str) -> None:
        self.entered = entered
        self.proceed = proceed
        self.value = value

    def __deepcopy__(self, memo: dict[int, Any]) -> PausingPayload:
        self.entered.set()
        if not self.proceed.wait(timeout=2):
            raise RuntimeError("copy pause timed out")
        return PausingPayload(self.entered, self.proceed, self.value)


def test_payloads_are_detached_on_ingress_and_every_egress() -> None:
    index = BoxIndex(2)
    payload = {"nested": [1]}
    handle = index.insert(Box((0, 0), (2, 2)), payload)
    payload["nested"].append(2)

    queried = index.get(handle)
    listed = index.entries()[0]
    matched = index.overlaps(Box((1, 1), (2, 2)))[0]
    snapshot = index.snapshot()
    for entry in (queried, listed, matched, snapshot.entries[0]):
        entry.data["nested"].append(99)

    expected = {"nested": [1]}
    assert index.get(handle).data == expected


def test_insert_update_remove_and_read_copy_failures_are_atomic() -> None:
    index = BoxIndex(2)
    box = Box((0, 0), (2, 2))
    controller = CopyController()
    controller.fail_at = 1
    with pytest.raises(RuntimeError, match="copy failed"):
        index.insert(box, ControlledPayload(controller, "failed insert"))
    assert index.diagnostics().version == 0

    controller.fail_at = None
    handle = index.insert(box, ControlledPayload(controller, "stored"))
    assert handle.sequence == 1
    before = index.diagnostics()

    controller.fail_at = controller.calls + 1
    with pytest.raises(RuntimeError, match="copy failed"):
        index.update(handle, data=ControlledPayload(controller, "replacement"))
    assert index.diagnostics() == before

    controller.fail_at = controller.calls + 1
    with pytest.raises(RuntimeError, match="copy failed"):
        index.get(handle)
    assert index.diagnostics() == before

    controller.fail_at = controller.calls + 1
    with pytest.raises(RuntimeError, match="copy failed"):
        index.remove(handle)
    assert index.diagnostics() == before

    controller.fail_at = None
    assert index.get(handle).data.value == "stored"
    assert index.remove(handle).handle == handle


def test_update_second_copy_failure_does_not_commit_candidate() -> None:
    controller = CopyController()
    index = BoxIndex(2)
    handle = index.insert(Box((0, 0), (1, 1)), ControlledPayload(controller, "stored"))
    before = index.diagnostics()
    controller.fail_at = controller.calls + 2

    with pytest.raises(RuntimeError, match="copy failed"):
        index.update(handle, data=ControlledPayload(controller, "replacement"))

    controller.fail_at = None
    assert index.diagnostics() == before
    assert index.get(handle).data.value == "stored"


def test_cross_thread_writer_waits_for_payload_cloner() -> None:
    entered = Event()
    proceed = Event()
    nested_done = Event()
    errors: list[BaseException] = []
    handles: list[int] = []

    def cloner(data: Any) -> Any:
        if data == "trigger":
            entered.set()
            if not proceed.wait(timeout=2):
                raise RuntimeError("copy pause timed out")
        return deepcopy(data)

    index = BoxIndex(2, payload_cloner=cloner)

    def insert(box: Box, data: str) -> None:
        try:
            handles.append(index.insert(box, data).sequence)
            if data == "nested":
                nested_done.set()
        except BaseException as exc:  # capture worker evidence
            errors.append(exc)

    first = Thread(target=insert, args=(Box((0, 0), (1, 1)), "trigger"))
    nested = Thread(target=insert, args=(Box((5, 5), (6, 6)), "nested"))
    first.start()
    assert entered.wait(timeout=2)
    nested.start()
    assert not nested_done.wait(timeout=0.05)
    proceed.set()
    first.join(timeout=2)
    nested.join(timeout=2)

    assert not errors
    assert handles == [1, 2]
    assert len(index) == 2


def test_snapshot_cloner_cannot_mutate_the_live_index() -> None:
    armed = False
    index: BoxIndex

    def cloner(data: Any) -> Any:
        if armed:
            index.insert(Box((5, 5), (6, 6)), "nested")
        return deepcopy(data)

    index = BoxIndex(2, payload_cloner=cloner)
    handle = index.insert(Box((0, 0), (1, 1)), "stored")
    snapshot = index.snapshot()
    armed = True

    with pytest.raises(RuntimeError, match="payload copying"):
        snapshot.get(handle)

    assert len(index) == 1
    assert index.diagnostics().version == 1


def test_same_index_mutation_from_deepcopy_is_rejected_without_sequence_drift() -> None:
    index = BoxIndex(2)
    box = Box((0, 0), (1, 1))

    class ReentrantPayload:
        def __deepcopy__(self, memo: dict[int, Any]) -> ReentrantPayload:
            index.insert(Box((5, 5), (6, 6)), "nested")
            return self

    with pytest.raises(RuntimeError, match="payload copying"):
        index.insert(box, ReentrantPayload())

    handle = index.insert(box, "safe")
    assert handle.sequence == 1
    assert index.diagnostics().version == 1


def test_reads_observe_only_complete_state_while_update_copy_is_paused() -> None:
    index = BoxIndex(2)
    handle = index.insert(Box((0, 0), (1, 1)), "before")
    entered = Event()
    proceed = Event()
    update_done = Event()
    read_done = Event()
    observed: list[str] = []

    def update() -> None:
        index.update(handle, data=PausingPayload(entered, proceed, "after"))
        update_done.set()

    def read() -> None:
        observed.append(index.get(handle).data.value)
        read_done.set()

    updater = Thread(target=update)
    reader = Thread(target=read)
    updater.start()
    assert entered.wait(timeout=2)
    reader.start()
    assert not read_done.wait(timeout=0.05)
    proceed.set()
    updater.join(timeout=2)
    reader.join(timeout=2)

    assert update_done.is_set()
    assert read_done.is_set()
    assert observed == ["after"]


def test_snapshot_preserves_aliases_internally_but_not_with_live_index() -> None:
    index = BoxIndex(2)
    shared = [1]
    first = index.insert(Box((0, 0), (1, 1)), {"value": shared})
    second = index.insert(Box((2, 2), (3, 3)), {"value": shared})

    # Separate insert calls intentionally create separate owned payload graphs.
    snapshot = index.snapshot()
    snapshot.entries[0].data["value"].append(2)
    assert snapshot.entries[1].data["value"] == [1]
    assert index.get(first).data["value"] == [1]
    assert index.get(second).data["value"] == [1]

    detached = deepcopy(snapshot)
    detached.entries[0].data["value"].append(3)
    assert snapshot.entries[0].data["value"] == [1, 2]
