from __future__ import annotations

from dataclasses import replace

import pytest

import treemendous.backends.probe as probe_module
from treemendous.backends.catalog import CATALOG_BY_ID
from treemendous.backends.types import (
    Algorithm,
    Available,
    BackendSpec,
    Capability,
    Device,
    Invalid,
    Maturity,
    Runtime,
    Unavailable,
)
from treemendous.basic.boundary import IntervalManager
from treemendous.domain import Span


def _spec(
    implementation: type[object],
    *,
    backend_id: str = "fake",
    runtime: Runtime = Runtime.PYTHON,
    maturity: Maturity = Maturity.STABLE,
    capabilities: frozenset[Capability] = frozenset({Capability.CORE}),
) -> BackendSpec:
    return BackendSpec(
        id=backend_id,
        name=backend_id,
        algorithm=Algorithm.BOUNDARY,
        runtime=runtime,
        device=Device.CPU,
        maturity=maturity,
        capabilities=capabilities,
        coordinate_bits=64,
        deterministic=True,
        loader=lambda: implementation,
        constructor_args={},
    )


@pytest.mark.parametrize(
    "backend_id",
    [
        "py_boundary",
        "py_avl_earliest",
        "py_summary",
        "py_treap",
        "py_boundary_summary",
    ],
)
def test_probe_validates_every_declared_python_capability(backend_id: str) -> None:
    spec = CATALOG_BY_ID[backend_id]
    state = probe_module.probe_backend(spec)
    assert isinstance(state, Available)
    assert state.validated_capabilities == spec.capabilities


def test_experimental_probe_short_circuits_without_loading() -> None:
    class MustNotLoad:
        def __init__(self) -> None:
            raise AssertionError("experimental loader was called")

    state = probe_module.probe_backend(
        _spec(MustNotLoad, maturity=Maturity.EXPERIMENTAL)
    )
    assert isinstance(state, Unavailable)
    assert "experimental" in state.reason


def test_loader_failure_is_unavailable_not_semantically_invalid() -> None:
    class BrokenLoader:
        def __init__(self) -> None:
            raise ImportError("dependency absent")

    state = probe_module.probe_backend(_spec(BrokenLoader))
    assert isinstance(state, Unavailable)
    assert "dependency absent" in state.reason


class _InitiallyNonEmpty(IntervalManager):
    def __init__(self) -> None:
        super().__init__()
        self.release_interval(0, 1)


class _AcceptsInvalidRelease(IntervalManager):
    def release_interval(self, start: int, end: int) -> None:
        if start == end:
            return
        super().release_interval(start, end)


class _MutatesBeforeInvalidRelease(IntervalManager):
    def release_interval(self, start: int, end: int) -> None:
        if start == end:
            super().release_interval(0, 1)
            raise ValueError("invalid after mutation")
        super().release_interval(start, end)


class _NeverFits(IntervalManager):
    def find_interval(self, start: int, length: int):
        return None


class _IgnoresValidReserve(IntervalManager):
    def reserve_interval(self, start: int, end: int) -> None:
        if start == end:
            Span(start, end)
        return None


@pytest.mark.parametrize(
    ("implementation", "message"),
    [
        (_InitiallyNonEmpty, "new implementation is not empty"),
        (_AcceptsInvalidRelease, "must reject invalid spans"),
        (_MutatesBeforeInvalidRelease, "changed observable state"),
        (_NeverFits, "fragmented first-fit"),
        (_IgnoresValidReserve, "reserve snapshot"),
    ],
)
def test_core_probe_rejects_observable_semantic_faults(
    implementation: type[object], message: str
) -> None:
    state = probe_module.probe_backend(_spec(implementation))
    assert isinstance(state, Invalid)
    assert message in state.error


@pytest.mark.parametrize(
    ("capability", "message"),
    [
        (Capability.ANALYTICS, "analytics API is absent"),
        (Capability.BEST_FIT, "best-fit API is absent"),
        (Capability.RANDOM_SAMPLE, "random-sampling API is absent"),
    ],
)
def test_capability_probe_rejects_declared_but_absent_interfaces(
    capability: Capability, message: str
) -> None:
    spec = _spec(
        IntervalManager,
        capabilities=frozenset({Capability.CORE, capability}),
    )
    state = probe_module.probe_backend(spec)
    assert isinstance(state, Invalid)
    assert message in state.error


def test_cpp_runtime_probe_requires_checked_int64_overflow() -> None:
    state = probe_module.probe_backend(_spec(IntervalManager, runtime=Runtime.CPP))
    assert isinstance(state, Invalid)
    assert "aggregate overflow" in state.error


def test_declared_capability_without_semantic_probe_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(probe_module._CAPABILITY_PROBES, Capability.ANALYTICS)
    spec = replace(
        CATALOG_BY_ID["py_boundary"],
        capabilities=frozenset({Capability.CORE, Capability.ANALYTICS}),
    )
    state = probe_module.probe_backend(spec)
    assert isinstance(state, Invalid)
    assert "no semantic probe for ANALYTICS" in state.error
