"""Exact contracts for the geometry-only ``RangeSet`` snapshot cache."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

import pytest

from treemendous import (
    IntervalResult,
    JoinPayloadPolicy,
    OrderedPayloadPolicy,
    RangeSnapshot,
    Span,
    UniformPayloadPolicy,
    create_range_set,
)
from treemendous.backends import CATALOG, Available, Maturity, probe_backend
from treemendous.backends.adapters import BackendAdapter
from treemendous.basic.boundary import IntervalManager
from treemendous.rangeset import RangeSet


class _NonAuthoritativeManager:
    def __init__(self) -> None:
        self._manager = IntervalManager()

    def release_interval(self, start: int, end: int) -> None:
        self._manager.release_interval(start, end)

    def reserve_interval(self, start: int, end: int) -> None:
        self._manager.reserve_interval(start, end)

    def get_intervals(self) -> list[IntervalResult]:
        return self._manager.get_intervals()


def test_geometry_snapshot_is_exact_complete_and_reused_until_change() -> None:
    ranges = create_range_set(
        ((0, 20), (30, 40)),
        backend="py_boundary",
        initially_available=False,
    )
    ranges.add(Span(10, 15))
    ranges.add(Span(2, 6))

    first = ranges.snapshot()
    second = ranges.snapshot()
    assert type(first) is RangeSnapshot
    assert first is second
    assert ranges._snapshot_cache is first
    assert not hasattr(ranges, "_snapshot_version")
    assert first == RangeSnapshot(
        (
            IntervalResult(2, 6),
            IntervalResult(10, 15),
        ),
        9,
        ranges.domain,
    )
    assert first.intervals == ranges.intervals()
    assert first.total_free == sum(item.end - item.start for item in first.intervals)
    assert first.domain is ranges.domain

    changed = ranges.discard(Span(3, 5), require_covered=True)
    assert changed.changed == (Span(3, 5),)
    third = ranges.snapshot()
    assert third is ranges.snapshot()
    assert third is not first
    assert third.intervals == (
        IntervalResult(2, 3),
        IntervalResult(5, 6),
        IntervalResult(10, 15),
    )

    # Previously published snapshots remain immutable point-in-time values.
    assert first.intervals == (IntervalResult(2, 6), IntervalResult(10, 15))
    assert first.total_free == 9


def test_non_authoritative_geometry_path_reuses_and_invalidates_exactly() -> None:
    ranges = RangeSet(
        BackendAdapter(_NonAuthoritativeManager()),
        domain=(0, 20),
        initially_available=False,
    )
    ranges.add(Span(2, 8))
    cached = ranges.snapshot()
    assert cached is ranges.snapshot()
    assert not ranges.add(Span(3, 5)).changed
    assert cached is ranges.snapshot()

    ranges.discard(Span(4, 6), require_covered=True)
    changed = ranges.snapshot()
    assert changed is ranges.snapshot()
    assert changed is not cached
    assert changed.intervals == (IntervalResult(2, 4), IntervalResult(6, 8))
    assert cached.intervals == (IntervalResult(2, 8),)


def test_noop_rejected_discard_and_failed_allocate_retain_cached_snapshot() -> None:
    ranges = create_range_set((0, 20), backend="py_boundary", initially_available=False)
    ranges.add(Span(4, 8))
    cached = ranges.snapshot()

    assert not ranges.add(Span(5, 7)).changed
    assert ranges.snapshot() is cached
    assert not ranges.discard(Span(10, 12)).changed
    assert ranges.snapshot() is cached
    rejected = ranges.discard(Span(2, 6), require_covered=True)
    assert rejected.changed == () and not rejected.fully_covered
    assert ranges.snapshot() is cached
    assert ranges.allocate(10, not_before=0) is None
    assert ranges.snapshot() is cached

    allocated = ranges.allocate(2, not_before=4)
    assert allocated == IntervalResult(4, 6)
    assert ranges.snapshot() is not cached


@pytest.mark.parametrize("domain", (None, (0, 20), ((0, 5), (10, 20))))
@pytest.mark.parametrize("initially_available", (False, True))
def test_domain_and_initial_availability_variants_cache_exact_snapshots(
    domain: Any, initially_available: bool
) -> None:
    ranges = create_range_set(
        domain,
        backend="py_boundary",
        initially_available=initially_available,
    )
    first = ranges.snapshot()
    assert type(first) is RangeSnapshot
    assert first is ranges.snapshot()
    assert first.domain == ranges.domain
    assert first.intervals == ranges.intervals()
    assert first.total_free == sum(item.end - item.start for item in first.intervals)


class _MutablePolicy:
    def can_merge(self, left: list[str], right: list[str]) -> bool:
        return left == right

    def combine(self, left: list[str], right: list[str]) -> list[str]:
        return left + right

    def restrict(self, data: list[str], source: Span, target: Span) -> list[str]:
        assert source.contains(target)
        return data


@pytest.mark.parametrize(
    ("policy", "payload"),
    (
        (UniformPayloadPolicy[list[str]](), ["uniform"]),
        (
            JoinPayloadPolicy[list[str]](
                lambda left, right: left + right,
                [],
            ),
            ["join"],
        ),
        (
            OrderedPayloadPolicy[list[str]](
                lambda left, right: left + right,
                [],
                event_key_fn=lambda value: tuple(value),
            ),
            ["ordered"],
        ),
        (_MutablePolicy(), ["custom"]),
    ),
    ids=("uniform", "join", "ordered", "custom"),
)
def test_payload_snapshots_remain_distinct_and_deeply_detached(
    policy: Any, payload: list[str]
) -> None:
    ranges = create_range_set(
        (0, 10),
        backend="py_boundary",
        initially_available=False,
        payload_policy=policy,
        payload_cloner=copy.deepcopy,
    )
    ranges.add(Span(2, 6), payload)
    first = ranges.snapshot()
    second = ranges.snapshot()
    assert type(first) is RangeSnapshot
    assert type(second) is RangeSnapshot
    assert first == second
    assert first is not second
    assert first.intervals is not second.intervals
    assert first.intervals[0] is not second.intervals[0]
    assert first.intervals[0].data is not second.intervals[0].data

    payload.append("caller mutation")
    first.intervals[0].data.append("snapshot mutation")
    assert ranges.snapshot().intervals[0].data not in (
        payload,
        first.intervals[0].data,
    )


def test_concurrent_snapshot_and_mutation_publications_are_consistent() -> None:
    ranges = create_range_set(
        (0, 200), backend="py_boundary", initially_available=False
    )
    for index in range(50):
        ranges.add(Span(index * 4, index * 4 + 2))
    start = Barrier(5)

    def reader() -> tuple[RangeSnapshot, ...]:
        start.wait()
        observed = []
        for _ in range(250):
            snapshot = ranges.snapshot()
            assert type(snapshot) is RangeSnapshot
            assert snapshot.total_free == sum(
                item.end - item.start for item in snapshot.intervals
            )
            assert all(
                left.end < right.start
                for left, right in zip(
                    snapshot.intervals, snapshot.intervals[1:], strict=False
                )
            )
            observed.append(snapshot)
        return tuple(observed)

    def writer() -> None:
        start.wait()
        target = Span(0, 2)
        for _ in range(125):
            assert ranges.discard(target, require_covered=True).changed
            assert ranges.add(target).changed

    with ThreadPoolExecutor(max_workers=5) as pool:
        readers = [pool.submit(reader) for _ in range(4)]
        mutation = pool.submit(writer)
        histories = [future.result(timeout=10) for future in readers]
        mutation.result(timeout=10)

    assert all(history for history in histories)
    assert ranges.snapshot().total_free == 100


@pytest.mark.parametrize(
    "spec",
    tuple(spec for spec in CATALOG if spec.maturity is Maturity.STABLE),
    ids=lambda spec: spec.id,
)
def test_cache_semantics_hold_for_every_available_stable_backend(spec: Any) -> None:
    state = probe_backend(spec)
    if not isinstance(state, Available):
        pytest.fail(f"stable backend {spec.id} unavailable: {state}")
    ranges = RangeSet(
        BackendAdapter(spec.loader()(**dict(spec.constructor_args))),
        domain=(0, 30),
        initially_available=False,
    )
    ranges.add(Span(2, 5))
    ranges.add(Span(10, 15))
    cached = ranges.snapshot()
    assert type(cached) is RangeSnapshot
    assert ranges.snapshot() is cached
    assert not ranges.add(Span(3, 4)).changed
    assert ranges.snapshot() is cached
    ranges.discard(Span(11, 13), require_covered=True)
    changed = ranges.snapshot()
    assert changed is ranges.snapshot()
    assert changed is not cached
    assert changed == RangeSnapshot(
        (
            IntervalResult(2, 5),
            IntervalResult(10, 11),
            IntervalResult(13, 15),
        ),
        6,
        ranges.domain,
    )
