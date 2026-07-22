"""Contracts for the collapsed authoritative hot path and scalar mutators.

These cover the selectable locking level, the fully-native scalar
``release``/``reserve`` surface, and the exact geometry equivalence between the
``MutationResult`` (``add``/``discard``) and scalar paths across every
authoritative stable backend.
"""

from __future__ import annotations

import dataclasses
from threading import RLock, Thread
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from treemendous.cpp.boundary import IntervalManager  # type: ignore[import-not-found]

from treemendous import Span, create_range_set
from treemendous.backends.adapters import BackendAdapter
from treemendous.backends.catalog import CATALOG_BY_ID
from treemendous.domain import (
    IntervalResult,
    MutationResult,
    _native_mutation_result,
    _native_span,
)
from treemendous.policies import UniformPayloadPolicy
from treemendous.rangeset import RangeSet, _NoOpLock

MIN_I64 = -(2**63)
MAX_I64 = 2**63 - 1

# Authoritative stable backends: cpp_boundary exposes the native scalar
# mutators; the Python authoritative backends fall back to the *_with_delta
# result's changed_length, which must stay exact.
AUTHORITATIVE_BACKENDS = ("py_boundary", "py_boundary_summary", "cpp_boundary")
NON_AUTHORITATIVE_BACKENDS = ("py_avl_earliest", "py_summary", "py_treap")

_Op = tuple[str, int, int]

_OPS = st.lists(
    st.tuples(
        st.sampled_from(("add", "discard", "discard_rc")),
        st.integers(min_value=-(2**20), max_value=2**20),
        st.integers(min_value=1, max_value=128),
    ).map(lambda item: (item[0], item[1], item[1] + item[2])),
    max_size=60,
)


def _unbounded(backend: str) -> RangeSet:
    """Build a domain-free ``RangeSet`` so any coordinate is admissible."""
    spec = CATALOG_BY_ID[backend]
    implementation = spec.loader()(**dict(spec.constructor_args))
    return RangeSet(BackendAdapter(implementation), initially_available=False)


def _apply_result(manager: RangeSet, op: _Op) -> int:
    kind, start, end = op
    span = Span(start, end)
    if kind == "add":
        return manager.add(span).changed_length
    return manager.discard(span, require_covered=kind == "discard_rc").changed_length


def _apply_scalar(manager: RangeSet, op: _Op) -> int:
    kind, start, end = op
    span = Span(start, end)
    if kind == "add":
        return manager.release(span)
    return manager.reserve(span, require_covered=kind == "discard_rc")


@pytest.mark.parametrize("backend", AUTHORITATIVE_BACKENDS)
@given(operations=_OPS)
@settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_scalar_matches_mutation_result_across_authoritative_backends(
    backend: str, operations: list[_Op]
) -> None:
    result_twin = _unbounded(backend)
    scalar_twin = _unbounded(backend)
    for op in operations:
        changed_result = _apply_result(result_twin, op)
        changed_scalar = _apply_scalar(scalar_twin, op)
        assert changed_result == changed_scalar
        assert result_twin.snapshot() == scalar_twin.snapshot()


@pytest.mark.parametrize("backend", AUTHORITATIVE_BACKENDS)
def test_scalar_covers_boundaries_negatives_and_no_ops(backend: str) -> None:
    result_twin = _unbounded(backend)
    scalar_twin = _unbounded(backend)
    script: list[_Op] = [
        ("add", -50, -10),  # negative coordinates
        ("add", -10, 20),  # adjacency merge across zero
        ("discard", -30, -20),  # interior split
        ("discard_rc", -100, -60),  # rejected: not fully covered
        ("discard_rc", -50, -40),  # accepted: fully covered
        ("add", -50, -40),  # exact restoration
        ("discard", -200, -150),  # no-op discard of unheld region
        ("add", -50, -10),  # no-op add of already-free region
    ]
    for op in script:
        assert _apply_result(result_twin, op) == _apply_scalar(scalar_twin, op)
        assert result_twin.snapshot() == scalar_twin.snapshot()


def test_scalar_matches_mutation_result_at_int64_extremes_on_cpp_boundary() -> None:
    result_twin = RangeSet(BackendAdapter(IntervalManager()), initially_available=False)
    scalar_twin = RangeSet(BackendAdapter(IntervalManager()), initially_available=False)
    script: list[_Op] = [
        ("add", MIN_I64, MIN_I64 + 16),
        ("add", MAX_I64 - 16, MAX_I64),
        ("discard", MIN_I64, MIN_I64 + 4),
        ("discard_rc", MAX_I64 - 32, MAX_I64),  # not fully covered -> reject
        ("discard_rc", MAX_I64 - 16, MAX_I64),  # fully covered -> accept
        ("add", MAX_I64 - 16, MAX_I64),
    ]
    for op in script:
        assert _apply_result(result_twin, op) == _apply_scalar(scalar_twin, op)
        assert result_twin.snapshot() == scalar_twin.snapshot()


# --- Locking level -----------------------------------------------------------


def test_default_is_synchronized_with_a_reentrant_lock() -> None:
    ranges = create_range_set((0, 10), backend="cpp_boundary")
    assert ranges.synchronized is True
    assert isinstance(ranges._lock, RLock().__class__)


def test_unsynchronized_installs_a_no_op_lock_sentinel() -> None:
    ranges = create_range_set((0, 10), backend="cpp_boundary", synchronized=False)
    assert ranges.synchronized is False
    assert isinstance(ranges._lock, _NoOpLock)
    # The no-op lock is a working context manager and acquire/release surface.
    with ranges._lock:
        assert ranges._lock.acquire(timeout=0.01) is True
        ranges._lock.release()


def test_synchronized_and_unsynchronized_agree_single_threaded() -> None:
    script: list[_Op] = [
        ("add", 0, 40),
        ("discard", 8, 16),
        ("discard_rc", 30, 60),
        ("reserve_scalar", 20, 24),
        ("release_scalar", 20, 24),
    ]
    sync = create_range_set((0, 64), backend="cpp_boundary", initially_available=False)
    unsync = create_range_set(
        (0, 64), backend="cpp_boundary", initially_available=False, synchronized=False
    )
    for kind, start, end in script:
        span = Span(start, end)
        if kind == "add":
            sync.add(span)
            unsync.add(span)
        elif kind.startswith("discard"):
            rc = kind == "discard_rc"
            sync.discard(span, require_covered=rc)
            unsync.discard(span, require_covered=rc)
        elif kind == "reserve_scalar":
            assert sync.reserve(span) == unsync.reserve(span)
        else:
            assert sync.release(span) == unsync.release(span)
    assert sync.snapshot() == unsync.snapshot()


@pytest.mark.parametrize("synchronized", [True, False])
@pytest.mark.parametrize("method", ["discard", "reserve"])
def test_both_paths_honor_the_selected_lock_and_reentrancy_guard(
    synchronized: bool, method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    ranges = create_range_set(
        (0, 10), backend="cpp_boundary", synchronized=synchronized
    )
    expected = _NoOpLock if not synchronized else RLock().__class__
    assert isinstance(ranges._lock, expected)

    # Prove the reentrancy guard fires regardless of the lock: a nested mutation
    # from inside a running one must be rejected.
    attempted = False
    if method == "discard":
        original = ranges._native_reserve
        assert original is not None

        def reenter(start: int, end: int, require_covered: bool) -> MutationResult:
            nonlocal attempted
            if not attempted:
                attempted = True
                with pytest.raises(RuntimeError, match="reentrant mutation"):
                    ranges.discard(Span(0, 1))
            return original(start, end, require_covered)

        monkeypatch.setattr(ranges, "_native_reserve", reenter)
        ranges.discard(Span(2, 4))
    else:
        scalar = ranges._native_scalar_reserve
        assert scalar is not None

        def reenter_scalar(start: int, end: int, require_covered: bool) -> int:
            nonlocal attempted
            if not attempted:
                attempted = True
                with pytest.raises(RuntimeError, match="reentrant mutation"):
                    ranges.reserve(Span(0, 1))
            return scalar(start, end, require_covered)

        monkeypatch.setattr(ranges, "_native_scalar_reserve", reenter_scalar)
        ranges.reserve(Span(2, 4))
    assert attempted


def test_synchronized_scalar_mutations_are_concurrently_consistent() -> None:
    ranges = create_range_set(
        (0, 4096), backend="cpp_boundary", initially_available=True
    )
    before = ranges.snapshot()

    def worker(offset: int) -> None:
        for index in range(200):
            base = offset + (index % 8) * 256
            span = Span(base, base + 32)
            ranges.reserve(span)
            ranges.release(span)  # restorative

    threads = [Thread(target=worker, args=(worker_id * 8,)) for worker_id in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert ranges.snapshot() == before
    assert ranges.snapshot().total_free == 4096


# --- Atomicity ---------------------------------------------------------------


def _snapshot_pair(ranges: RangeSet) -> tuple[Any, int]:
    snapshot = ranges.snapshot()
    return snapshot.intervals, snapshot.total_free


def test_collapsed_add_discard_are_atomic_on_result_construction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ranges = RangeSet(BackendAdapter(IntervalManager()), initially_available=False)
    ranges.add(Span(0, 10))
    before = _snapshot_pair(ranges)

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected result failure")

    monkeypatch.setattr("treemendous.domain._object_new", fail)
    with pytest.raises(RuntimeError, match="injected result failure"):
        ranges.discard(Span(2, 4))
    with pytest.raises(RuntimeError, match="injected result failure"):
        ranges.add(Span(20, 24))
    monkeypatch.undo()
    assert _snapshot_pair(ranges) == before


def test_scalar_fallback_is_atomic_on_result_construction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force the *_with_delta fallback (as a Python authoritative backend uses)
    # by disabling the native scalar methods, then prove result-construction
    # failure leaves state untouched.
    ranges = RangeSet(BackendAdapter(IntervalManager()), initially_available=False)
    ranges.add(Span(0, 10))
    monkeypatch.setattr(ranges, "_native_scalar_release", None)
    monkeypatch.setattr(ranges, "_native_scalar_reserve", None)
    before = _snapshot_pair(ranges)

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected result failure")

    monkeypatch.setattr("treemendous.domain._object_new", fail)
    with pytest.raises(RuntimeError, match="injected result failure"):
        ranges.reserve(Span(2, 4))
    with pytest.raises(RuntimeError, match="injected result failure"):
        ranges.release(Span(20, 24))
    monkeypatch.undo()
    assert _snapshot_pair(ranges) == before


def test_scalar_and_collapsed_paths_are_atomic_on_domain_and_overflow() -> None:
    domained = create_range_set((0, 10), backend="cpp_boundary")
    before = _snapshot_pair(domained)
    with pytest.raises(ValueError, match="managed domain"):
        domained.reserve(Span(5, 20))
    with pytest.raises(ValueError, match="managed domain"):
        domained.release(Span(5, 20))
    with pytest.raises(ValueError, match="managed domain"):
        domained.discard(Span(5, 20))
    assert _snapshot_pair(domained) == before

    unbounded = RangeSet(BackendAdapter(IntervalManager()), initially_available=False)
    unbounded.release(Span(MIN_I64, -1))
    before = _snapshot_pair(unbounded)
    with pytest.raises(OverflowError):
        unbounded.release(Span(-1, MAX_I64))  # aggregate overflow before apply
    with pytest.raises(OverflowError):
        unbounded.add(Span(-1, MAX_I64))
    assert _snapshot_pair(unbounded) == before


# --- Capability guards -------------------------------------------------------


def test_scalar_rejects_payload_policy() -> None:
    ranges = create_range_set(
        (0, 10), backend="py_boundary", payload_policy=UniformPayloadPolicy()
    )
    with pytest.raises(ValueError, match="payload policy"):
        ranges.reserve(Span(0, 2))
    with pytest.raises(ValueError, match="payload policy"):
        ranges.release(Span(0, 2))


@pytest.mark.parametrize("backend", NON_AUTHORITATIVE_BACKENDS)
def test_scalar_rejects_non_authoritative_backends(backend: str) -> None:
    ranges = create_range_set((0, 10), backend=backend)
    before = _snapshot_pair(ranges)
    with pytest.raises(ValueError, match="authoritative geometry backend"):
        ranges.reserve(Span(0, 2))
    with pytest.raises(ValueError, match="authoritative geometry backend"):
        ranges.release(Span(0, 2))
    assert _snapshot_pair(ranges) == before


class _BadDeltaManager:
    """Claims authoritative geometry but violates the MutationResult contract."""

    _treemendous_authoritative_geometry = True

    def release_with_delta(self, start: int, end: int) -> tuple[int, int]:
        return (start, end)  # not a MutationResult

    def reserve_with_delta(self, start: int, end: int, require_covered: bool) -> None:
        return None

    def find_interval(self, start: int, length: int) -> None:
        return None

    def find_overlapping_intervals(self, start: int, end: int) -> list[tuple[int, int]]:
        return []

    def get_intervals(self) -> list[tuple[int, int]]:
        return []

    def release_interval(self, start: int, end: int) -> None:
        return None

    def reserve_interval(self, start: int, end: int) -> None:
        return None


def test_construction_probe_fails_closed_without_a_mutation_result() -> None:
    adapter = BackendAdapter(_BadDeltaManager())
    assert adapter.supports_authoritative_geometry is True
    with pytest.raises(TypeError, match="must return MutationResult"):
        RangeSet(adapter, initially_available=False)


# --- Public API / type contracts ---------------------------------------------


def test_public_result_types_remain_exact_frozen_dataclasses() -> None:
    for cls in (Span, IntervalResult, MutationResult):
        assert dataclasses.is_dataclass(cls)
        assert getattr(cls, "__dataclass_params__").frozen is True

    native = _native_span(2, 5)
    validated = Span(2, 5)
    assert native == validated
    assert hash(native) == hash(validated)
    assert native.length == 3

    native_result = _native_mutation_result((native,), 3, False)
    validated_result = MutationResult((validated,), 3, False)
    assert native_result == validated_result

    # User Span construction still validates.
    with pytest.raises(ValueError, match="start < end"):
        Span(5, 2)
    with pytest.raises(TypeError, match="must be an integer"):
        Span(1.0, 2)  # type: ignore[arg-type]


@pytest.mark.parametrize("synchronized", [True, False])
def test_scalar_mutations_invalidate_the_geometry_snapshot_cache(
    synchronized: bool,
) -> None:
    """The scalar surface must invalidate the geometry-only snapshot cache.

    ``snapshot()`` caches an immutable geometry-only snapshot and reuses it while
    the geometry is unchanged.  The scalar ``reserve``/``release`` mutators must
    invalidate that cache on a changed mutation (and preserve it on a no-op),
    exactly as ``add``/``discard`` do.
    """
    ranges = create_range_set(
        domain=(0, 100),
        backend="cpp_boundary",
        initially_available=True,
        synchronized=synchronized,
    )
    first = ranges.snapshot()
    assert ranges.snapshot() is first  # unchanged geometry reuses the cache

    assert ranges.reserve(Span(10, 20)) == 10
    after_reserve = ranges.snapshot()
    assert after_reserve is not first
    assert sum(iv.end - iv.start for iv in after_reserve.intervals) == 90

    cached = ranges.snapshot()
    assert cached is after_reserve  # unchanged again -> cached

    # A no-op scalar mutation must not swap out the still-valid cached snapshot.
    assert ranges.reserve(Span(10, 20), require_covered=True) == 0
    assert ranges.snapshot() is cached

    assert ranges.release(Span(10, 20)) == 10
    restored = ranges.snapshot()
    assert restored is not cached
    assert sum(iv.end - iv.start for iv in restored.intervals) == 100
