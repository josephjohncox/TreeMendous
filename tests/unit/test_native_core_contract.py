"""Checked-int64 and atomicity regressions for stable CPU extensions."""

from __future__ import annotations

import importlib

import pytest

from treemendous.domain import IntervalResult, MutationResult

MODULES = ("treemendous.cpp.boundary",)
MIN_I64 = -(2**63)
MAX_I64 = 2**63 - 1


def _manager(module_name: str):
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        pytest.fail(f"stable CPU extension {module_name} unavailable: {exc}")
    return module.IntervalManager()


def _snapshot(manager):
    return (tuple(manager.get_intervals()), manager.get_total_available_length())


def _expect_equal(actual, expected) -> None:
    if actual != expected:
        pytest.fail(f"expected {expected!r}, got {actual!r}")


def _changed_pairs(result) -> tuple[tuple[int, int], ...]:
    return tuple((span.start, span.end) for span in result.changed)


@pytest.mark.parametrize("module_name", MODULES)
def test_raw_no_fit_is_none(module_name: str) -> None:
    manager = _manager(module_name)
    manager.release_interval(0, 10)
    assert manager.find_interval(0, 50) is None


@pytest.mark.parametrize("module_name", MODULES)
def test_release_length_and_aggregate_overflow_are_atomic(module_name: str) -> None:
    manager = _manager(module_name)
    before = _snapshot(manager)
    with pytest.raises(OverflowError):
        manager.release_interval(MIN_I64, MAX_I64)
    assert _snapshot(manager) == before

    manager.release_interval(MIN_I64, -1)
    before = _snapshot(manager)
    with pytest.raises(OverflowError):
        manager.release_interval(-1, MAX_I64)
    assert _snapshot(manager) == before

    aggregate = _manager(module_name)
    aggregate.release_interval(MIN_I64, -1)
    before = _snapshot(aggregate)
    with pytest.raises(OverflowError):
        aggregate.release_interval(0, MAX_I64)
    assert _snapshot(aggregate) == before


@pytest.mark.parametrize("module_name", MODULES)
def test_native_result_construction_failure_is_atomic(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(module_name)
    manager.release_interval(0, 10)
    before = _snapshot(manager)

    def fail_result(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected result failure")

    monkeypatch.setattr(MutationResult, "__init__", fail_result)
    with pytest.raises(RuntimeError, match="injected result failure"):
        manager.reserve_with_delta(2, 4, False)
    _expect_equal(_snapshot(manager), before)

    monkeypatch.undo()
    monkeypatch.setattr(IntervalResult, "__init__", fail_result)
    with pytest.raises(RuntimeError, match="injected result failure"):
        manager.allocate_interval(0, 2, None)
    _expect_equal(_snapshot(manager), before)


@pytest.mark.parametrize("module_name", MODULES)
def test_native_mutation_deltas_are_exact_and_atomic(module_name: str) -> None:
    manager = _manager(module_name)
    manager.release_interval(0, 2)
    manager.release_interval(4, 6)

    released = manager.release_with_delta(1, 5)
    expected_released = ((2, 4),)
    assert _changed_pairs(released) == expected_released
    assert released.changed_length == 2
    assert not released.fully_covered
    _expect_equal(_snapshot(manager), (((0, 6),), 6))

    no_op = manager.release_with_delta(2, 4)
    assert not no_op.changed
    assert no_op.changed_length == 0
    assert no_op.fully_covered

    before = _snapshot(manager)
    rejected = manager.reserve_with_delta(1, 7, True)
    assert not rejected.changed
    assert rejected.changed_length == 0
    assert not rejected.fully_covered
    assert _snapshot(manager) == before

    reserved = manager.reserve_with_delta(1, 5, True)
    expected_reserved = ((1, 5),)
    assert _changed_pairs(reserved) == expected_reserved
    assert reserved.changed_length == 4
    assert reserved.fully_covered
    _expect_equal(_snapshot(manager), (((0, 1), (5, 6)), 2))

    expected_overlaps = [(0, 1), (5, 6)]
    assert manager.find_overlapping_intervals(0, 6) == expected_overlaps
    assert manager.get_interval_count() == 2
    assert manager.get_largest_available_length() == 1

    manager.release_interval(6, 12)
    allocated = manager.allocate_interval(5, 3, 10)
    _expect_equal((allocated.start, allocated.end), (5, 8))
    _expect_equal(_snapshot(manager), (((0, 1), (8, 12)), 5))
    before_bounded = _snapshot(manager)
    assert manager.allocate_interval(8, 3, 10) is None
    assert _snapshot(manager) == before_bounded

    before = _snapshot(manager)
    with pytest.raises(OverflowError):
        manager.release_with_delta(MIN_I64, MAX_I64)
    assert _snapshot(manager) == before


@pytest.mark.parametrize("module_name", MODULES)
def test_native_managed_domain_rejects_gap_crossing_atomically(
    module_name: str,
) -> None:
    manager = _manager(module_name)
    manager.set_managed_domain([(0, 5), (10, 15)])
    manager.release_interval(0, 5)
    before = _snapshot(manager)

    with pytest.raises(ValueError, match="managed domain"):
        manager.release_with_delta(5, 10)
    assert _snapshot(manager) == before
    with pytest.raises(ValueError, match="managed domain"):
        manager.reserve_with_delta(4, 11, False)
    assert _snapshot(manager) == before
    with pytest.raises(ValueError, match="managed domain"):
        manager.find_overlapping_intervals(4, 11)
    assert _snapshot(manager) == before
    with pytest.raises(RuntimeError, match="before mutation"):
        manager.set_managed_domain([(0, 20)])
    assert _snapshot(manager) == before


@pytest.mark.parametrize("module_name", MODULES)
def test_native_managed_domain_normalizes_and_checks_coordinates(
    module_name: str,
) -> None:
    manager = _manager(module_name)
    manager.set_managed_domain([(8, 10), (0, 5), (4, 8)])
    manager.release_interval(4, 9)
    _expect_equal(_snapshot(manager), (((4, 9),), 5))

    for spans in ([(False, 5)], [(0, True)]):
        fresh = _manager(module_name)
        with pytest.raises(TypeError, match="must be an integer"):
            fresh.set_managed_domain(spans)
        _expect_equal(_snapshot(fresh), ((), 0))

    fresh = _manager(module_name)
    with pytest.raises(OverflowError):
        fresh.set_managed_domain([(0, MAX_I64 + 1)])
    _expect_equal(_snapshot(fresh), ((), 0))


@pytest.mark.parametrize("module_name", MODULES)
def test_negative_coordinate_huge_length_does_not_underflow(module_name: str) -> None:
    manager = _manager(module_name)
    manager.release_interval(-10, -5)
    assert manager.find_interval(-10, MAX_I64) is None
    _expect_equal(_snapshot(manager), (((-10, -5),), 5))


@pytest.mark.parametrize("module_name", MODULES)
def test_binding_int64_conversion_uses_overflow_error_and_is_atomic(
    module_name: str,
) -> None:
    manager = _manager(module_name)
    manager.release_interval(0, 10)
    before = _snapshot(manager)
    for outside in (MIN_I64 - 1, MAX_I64 + 1):
        with pytest.raises(OverflowError):
            manager.release_interval(outside, 0)
        assert _snapshot(manager) == before
        with pytest.raises(OverflowError):
            manager.reserve_interval(outside, 0)
        assert _snapshot(manager) == before
        with pytest.raises(OverflowError):
            manager.find_interval(outside, 1)
        assert _snapshot(manager) == before


@pytest.mark.parametrize("module_name", MODULES)
def test_exact_int64_endpoints_and_checked_query_endpoint(module_name: str) -> None:
    manager = _manager(module_name)
    manager.release_interval(MIN_I64, -1)
    _expect_equal(manager.find_interval(MIN_I64, MAX_I64), (MIN_I64, -1))
    before = _snapshot(manager)
    assert manager.find_interval(MAX_I64 - 5, 10) is None
    assert _snapshot(manager) == before
