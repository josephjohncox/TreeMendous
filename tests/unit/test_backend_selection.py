from __future__ import annotations

from dataclasses import replace

import pytest

from treemendous.backends.selection import select_backend
from treemendous.backends.types import (
    Algorithm,
    Available,
    BackendRequest,
    BackendSpec,
    Capability,
    Device,
    Invalid,
    Maturity,
    Runtime,
    Unavailable,
)
from treemendous.domain import BackendUnavailableError


class _Backend:
    pass


def _spec(
    backend_id: str,
    *,
    runtime: Runtime = Runtime.PYTHON,
    maturity: Maturity = Maturity.STABLE,
    capabilities: frozenset[Capability] = frozenset({Capability.CORE}),
    coordinate_bits: int = 64,
    deterministic: bool = True,
) -> BackendSpec:
    return BackendSpec(
        id=backend_id,
        name=backend_id,
        algorithm=Algorithm.BOUNDARY,
        runtime=runtime,
        device=Device.CPU,
        maturity=maturity,
        capabilities=capabilities,
        coordinate_bits=coordinate_bits,
        deterministic=deterministic,
        loader=lambda: _Backend,
        constructor_args={},
    )


def _available(spec: BackendSpec) -> Available:
    return Available(spec.capabilities)


def test_selection_is_priority_driven_not_catalog_order() -> None:
    python = _spec("py_boundary")
    native = _spec("cpp_boundary", runtime=Runtime.CPP)
    summary = _spec("py_summary")
    specs = (python, summary, native)
    probes = {spec.id: _available(spec) for spec in specs}

    forward = select_backend(specs, probes, BackendRequest())
    reverse = select_backend(tuple(reversed(specs)), probes, BackendRequest())

    assert forward.selected is native
    assert reverse.selected is native
    assert not forward.rejected


def test_unknown_priority_preserves_catalog_order() -> None:
    first = _spec("custom-first")
    second = _spec("custom-second")
    probes = {spec.id: _available(spec) for spec in (first, second)}

    decision = select_backend((first, second), probes, BackendRequest())

    assert decision.selected is first


def test_selection_reports_every_rejection_reason_in_catalog_order() -> None:
    required = frozenset({Capability.CORE, Capability.ANALYTICS})
    good = _spec("good", capabilities=required)
    experimental = replace(
        _spec("experimental", capabilities=required), maturity=Maturity.EXPERIMENTAL
    )
    missing_probe = _spec("missing-probe", capabilities=required)
    unavailable = _spec("unavailable", capabilities=required)
    invalid = _spec("invalid", capabilities=required)
    missing_capability = _spec("missing-capability", capabilities=required)
    narrow = _spec("narrow", capabilities=required, coordinate_bits=32)
    random = _spec("random", capabilities=required, deterministic=False)
    wrong_runtime = _spec("wrong-runtime", capabilities=required, runtime=Runtime.CPP)
    specs = (
        experimental,
        missing_probe,
        unavailable,
        invalid,
        missing_capability,
        narrow,
        random,
        wrong_runtime,
        good,
    )
    probes = {
        experimental.id: _available(experimental),
        unavailable.id: Unavailable("library absent"),
        invalid.id: Invalid("semantic mismatch"),
        missing_capability.id: Available(frozenset({Capability.CORE})),
        narrow.id: _available(narrow),
        random.id: _available(random),
        wrong_runtime.id: _available(wrong_runtime),
        good.id: _available(good),
    }

    decision = select_backend(
        specs,
        probes,
        BackendRequest(require=required, preferred_runtime=Runtime.PYTHON),
    )

    expected = tuple(
        [
            ("experimental", "experimental"),
            ("missing-probe", "missing probe result"),
            ("unavailable", "library absent"),
            ("invalid", "semantic mismatch"),
            ("missing-capability", "missing required capability"),
            ("narrow", "coordinate width too small"),
            ("random", "not deterministic"),
            ("wrong-runtime", "runtime does not match preference"),
        ]
    )
    assert decision.selected is good
    assert decision.rejected == expected


def test_nondeterministic_backend_can_be_requested_explicitly() -> None:
    random = _spec("random", deterministic=False)
    decision = select_backend(
        (random,),
        {random.id: _available(random)},
        BackendRequest(deterministic=False),
    )
    assert decision.selected is random


def test_selection_fails_when_no_backend_satisfies_request() -> None:
    unavailable = _spec("unavailable")
    with pytest.raises(BackendUnavailableError, match="no backend satisfies"):
        select_backend(
            (unavailable,),
            {unavailable.id: Unavailable("not installed")},
            BackendRequest(),
        )
