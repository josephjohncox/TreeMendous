"""Strict conformance tests driven by the immutable production catalog."""

from __future__ import annotations

from typing import Any

import pytest

from treemendous.backends import (
    CATALOG,
    Available,
    Capability,
    Invalid,
    Maturity,
    Runtime,
    Unavailable,
    probe_backend,
)
from treemendous.backends.adapters import CppBackendAdapter, PythonBackendAdapter
from treemendous.domain import Span
from treemendous.policies import JoinPayloadPolicy
from treemendous.rangeset import RangeSet

STABLE_SPECS = tuple(spec for spec in CATALOG if spec.maturity is Maturity.STABLE)


def _adapter(spec: Any, implementation: Any):
    if spec.runtime is Runtime.PYTHON:
        return PythonBackendAdapter(implementation)
    return CppBackendAdapter(implementation)


def _expect_equal(actual: Any, expected: Any) -> None:
    if actual != expected:
        pytest.fail(f"expected {expected!r}, got {actual!r}")


@pytest.fixture(params=STABLE_SPECS, ids=lambda spec: spec.id)
def stable_rangeset(request: pytest.FixtureRequest) -> tuple[Any, RangeSet]:
    spec = request.param
    state = probe_backend(spec)
    if isinstance(state, Unavailable):
        pytest.fail(f"stable backend {spec.id} unavailable: {state.reason}")
    assert not isinstance(state, Invalid), f"{spec.id}: {state.error}"
    assert isinstance(state, Available)
    implementation = spec.loader()(**dict(spec.constructor_args))
    policy = (
        JoinPayloadPolicy(lambda left, right: left | right, frozenset())
        if Capability.PAYLOADS in spec.capabilities
        else None
    )
    ranges = RangeSet(
        _adapter(spec, implementation),
        domain=(0, 100),
        capabilities=state.validated_capabilities,
        initially_available=False,
        payload_policy=policy,
    )
    return spec, ranges


@pytest.mark.parametrize("spec", STABLE_SPECS, ids=lambda spec: spec.id)
def test_stable_catalog_probe_is_never_invalid(spec: Any) -> None:
    state = probe_backend(spec)
    assert not isinstance(state, Invalid), f"{spec.id}: {state.error}"
    assert not isinstance(state, Unavailable), (
        f"stable backend {spec.id}: {state.reason}"
    )
    assert isinstance(state, Available)
    assert state.validated_capabilities == spec.capabilities


def test_stable_state_sequence(stable_rangeset: tuple[Any, RangeSet]) -> None:
    _, ranges = stable_rangeset
    _expect_equal(ranges.snapshot().intervals, ())

    first = ranges.add(Span(0, 5))
    second = ranges.add(Span(10, 20))
    _expect_equal(first.changed, (Span(0, 5),))
    _expect_equal(second.changed, (Span(10, 20),))
    _expect_equal(
        [(item.start, item.end) for item in ranges.intervals()],
        [(0, 5), (10, 20)],
    )
    assert ranges.first_fit(8, not_before=0).span == Span(10, 18)
    assert ranges.first_fit(50, not_before=0) is None

    removed = ranges.discard(Span(12, 15))
    _expect_equal(removed.changed, (Span(12, 15),))
    assert removed.changed_length == 3
    _expect_equal(
        [(item.start, item.end) for item in ranges.intervals()],
        [(0, 5), (10, 12), (15, 20)],
    )
    assert ranges.snapshot().total_free == 12


def test_stable_invalid_mutations_are_atomic(
    stable_rangeset: tuple[Any, RangeSet],
) -> None:
    _, ranges = stable_rangeset
    ranges.add(Span(0, 20))
    before = ranges.snapshot()
    for start, end in ((1, 1), (9, 4)):
        with pytest.raises(ValueError):
            ranges.release_interval(start, end)
        assert ranges.snapshot() == before
        with pytest.raises(ValueError):
            ranges.reserve_interval(start, end)
        assert ranges.snapshot() == before


def test_stable_allocate_and_capabilities(
    stable_rangeset: tuple[Any, RangeSet],
) -> None:
    spec, ranges = stable_rangeset
    ranges.add(Span(0, 100))
    allocated = ranges.allocate(10, not_before=5, not_after=20)
    assert allocated is not None and allocated.span == Span(5, 15)
    assert ranges.snapshot().total_free == 90
    assert Capability.CORE in spec.capabilities
    assert Capability.ATOMIC_ALLOCATE in spec.capabilities


def test_payload_capability_geometry(stable_rangeset: tuple[Any, RangeSet]) -> None:
    spec, ranges = stable_rangeset
    if Capability.PAYLOADS not in spec.capabilities:
        pytest.skip(f"{spec.id}: payload capability not declared")
    ranges.add(Span(0, 10), frozenset({"A"}))
    ranges.add(Span(5, 15), frozenset({"B"}))
    _expect_equal(
        [(item.start, item.end, item.data) for item in ranges.intervals()],
        [
            (0, 5, frozenset({"A"})),
            (5, 10, frozenset({"A", "B"})),
            (10, 15, frozenset({"B"})),
        ],
    )
