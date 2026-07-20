"""Payload-cloner and locking contracts for optimized box indexes."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from threading import Event, Thread
from typing import Any

import pytest

from treemendous.multidimensional import (
    BoundedBoxIndex,
    Box,
    BoxIndex2D,
    BoxIndex3D,
    BoxIndex4D,
    BoxIndexProtocol,
)

ClonerFactory = Callable[[Callable[[Any], Any]], BoxIndexProtocol]


@pytest.fixture(
    params=[
        pytest.param(
            lambda cloner: BoxIndex2D(payload_cloner=cloner), id="projection-2d"
        ),
        pytest.param(
            lambda cloner: BoxIndex3D(payload_cloner=cloner), id="projection-3d"
        ),
        pytest.param(
            lambda cloner: BoxIndex4D(payload_cloner=cloner), id="projection-4d"
        ),
        pytest.param(
            lambda cloner: BoundedBoxIndex(
                Box((-10, -10), (10, 10)),
                (2, 2),
                payload_cloner=cloner,
            ),
            id="bounded-grid",
        ),
    ]
)
def cloner_factory(request: pytest.FixtureRequest) -> ClonerFactory:
    return request.param


def _box(index: BoxIndexProtocol, lower: int, upper: int) -> Box:
    return Box((lower,) * index.dimensions, (upper,) * index.dimensions)


def test_optimized_update_copy_failure_is_atomic(
    cloner_factory: ClonerFactory,
) -> None:
    calls = 0
    fail_at: int | None = None

    def cloner(value: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise RuntimeError("copy failed")
        return deepcopy(value)

    index = cloner_factory(cloner)
    handle = index.insert(_box(index, 0, 1), {"value": "stored"})
    before = index.snapshot()
    before_diagnostics = index.diagnostics()
    fail_at = calls + 1

    with pytest.raises(RuntimeError, match="copy failed"):
        index.update(
            handle,
            box=_box(index, 2, 3),
            data={"value": "replacement"},
        )

    fail_at = None
    assert index.snapshot() == before
    assert index.diagnostics() == before_diagnostics


def test_optimized_reads_wait_for_complete_update(
    cloner_factory: ClonerFactory,
) -> None:
    entered = Event()
    proceed = Event()
    read_done = Event()
    errors: list[BaseException] = []
    observed: list[str] = []

    def cloner(value: Any) -> Any:
        if value == "pause":
            entered.set()
            if not proceed.wait(timeout=2):
                raise RuntimeError("copy pause timed out")
        return deepcopy(value)

    index = cloner_factory(cloner)
    handle = index.insert(_box(index, 0, 1), "before")

    def update() -> None:
        try:
            index.update(handle, data="pause")
        except BaseException as exc:  # capture worker evidence
            errors.append(exc)

    def read() -> None:
        try:
            observed.append(index.get(handle).data)
            read_done.set()
        except BaseException as exc:  # capture worker evidence
            errors.append(exc)

    updater = Thread(target=update)
    reader = Thread(target=read)
    updater.start()
    assert entered.wait(timeout=2)
    reader.start()
    assert not read_done.wait(timeout=0.05)
    proceed.set()
    updater.join(timeout=2)
    reader.join(timeout=2)

    assert not errors
    assert not updater.is_alive()
    assert not reader.is_alive()
    assert observed == ["pause"]


def test_optimized_concurrent_writers_wait_instead_of_failing(
    cloner_factory: ClonerFactory,
) -> None:
    entered = Event()
    proceed = Event()
    second_done = Event()
    errors: list[BaseException] = []
    handles: list[int] = []

    def cloner(value: Any) -> Any:
        if value == "pause":
            entered.set()
            if not proceed.wait(timeout=2):
                raise RuntimeError("copy pause timed out")
        return deepcopy(value)

    index = cloner_factory(cloner)

    def insert(value: str) -> None:
        try:
            handles.append(index.insert(_box(index, 0, 1), value).sequence)
            if value == "second":
                second_done.set()
        except BaseException as exc:  # capture worker evidence
            errors.append(exc)

    first = Thread(target=insert, args=("pause",))
    second = Thread(target=insert, args=("second",))
    first.start()
    assert entered.wait(timeout=2)
    second.start()
    assert not second_done.wait(timeout=0.05)
    proceed.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not errors
    assert not first.is_alive()
    assert not second.is_alive()
    assert handles == [1, 2]
    assert [entry.data for entry in index.entries()] == ["pause", "second"]


def test_optimized_reentrant_mutation_is_rejected_without_sequence_drift(
    cloner_factory: ClonerFactory,
) -> None:
    index: BoxIndexProtocol

    class ReentrantPayload:
        def __deepcopy__(self, memo: dict[int, Any]) -> ReentrantPayload:
            index.insert(_box(index, 4, 5), "nested")
            return self

    index = cloner_factory(deepcopy)
    with pytest.raises(RuntimeError, match="payload copying"):
        index.insert(_box(index, 0, 1), ReentrantPayload())

    handle = index.insert(_box(index, 0, 1), "safe")
    assert handle.sequence == 1
    assert index.diagnostics().version == 1
