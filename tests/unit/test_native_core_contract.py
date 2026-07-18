"""Checked-int64 and atomicity regressions for stable CPU extensions."""

from __future__ import annotations

import importlib

import pytest

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
