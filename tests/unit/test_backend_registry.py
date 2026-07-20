from __future__ import annotations

from dataclasses import replace

import pytest

import treemendous
from treemendous.backends.catalog import CATALOG_BY_ID
from treemendous.backends.registry import BackendRegistry
from treemendous.backends.types import (
    Available,
    BackendRequest,
    Capability,
    Invalid,
    Runtime,
    Unavailable,
)
from treemendous.domain import BackendInvalidError, BackendUnavailableError, Span
from treemendous.policies import UniformPayloadPolicy


@pytest.fixture(scope="module")
def registry() -> BackendRegistry:
    return BackendRegistry.discover()


def _with_state(
    registry: BackendRegistry, backend_id: str, state: Available | Invalid | Unavailable
) -> BackendRegistry:
    states = dict(registry.states)
    states[backend_id] = state
    return BackendRegistry(registry.specs, states)


def test_registry_is_immutable_and_requires_exact_unique_state_keys(
    registry: BackendRegistry,
) -> None:
    with pytest.raises(TypeError):
        registry.states["py_boundary"] = Unavailable("changed")  # type: ignore[index]
    with pytest.raises(ValueError, match="exactly match"):
        BackendRegistry(registry.specs, {})
    duplicate = (*registry.specs, registry.specs[0])
    duplicate_states = {spec.id: registry.states[spec.id] for spec in duplicate}
    with pytest.raises(ValueError, match="unique"):
        BackendRegistry(duplicate, duplicate_states)


def test_registry_selection_is_typed_and_deterministic(
    registry: BackendRegistry,
) -> None:
    decision = registry.select()
    assert isinstance(decision.selected.id, str)
    assert isinstance(registry.states[decision.selected.id], Available)

    python = registry.select(BackendRequest(preferred_runtime=Runtime.PYTHON))
    assert python.selected.runtime is Runtime.PYTHON


def test_registry_creates_only_the_canonical_rangeset_interface(
    registry: BackendRegistry,
) -> None:
    ranges = registry.create(
        (0, 8),
        backend="py_boundary",
        initially_available=False,
        payload_policy=UniformPayloadPolicy(),
    )
    ranges.add(Span(0, 4), "cpu")
    assert ranges.first_fit(2, not_before=0).data == "cpu"
    assert not hasattr(ranges, "release_interval")
    assert not hasattr(ranges, "get_raw_implementation")

    public = treemendous.create_range_set(
        (0, 4), backend="py_boundary", initially_available=False
    )
    public.add(Span(1, 3))
    assert public.snapshot().total_free == 2


def test_registry_uses_validated_capabilities_for_requests(
    registry: BackendRegistry,
) -> None:
    restricted = _with_state(
        registry,
        "py_summary",
        Available(frozenset({Capability.CORE})),
    )
    with pytest.raises(BackendUnavailableError, match="no backend satisfies"):
        restricted.create(
            (0, 10),
            backend="py_summary",
            request=BackendRequest(require=frozenset({Capability.ANALYTICS})),
        )


def test_registry_distinguishes_unknown_invalid_and_unavailable(
    registry: BackendRegistry,
) -> None:
    with pytest.raises(BackendUnavailableError, match="unknown backend"):
        registry.create((0, 10), backend="missing")
    with pytest.raises(BackendUnavailableError, match="experimental"):
        registry.create((0, 10), backend="gpu_boundary_summary")

    invalid = _with_state(registry, "py_boundary", Invalid("broken law"))
    with pytest.raises(BackendInvalidError, match="broken law"):
        invalid.create((0, 10), backend="py_boundary")


def test_registry_reports_available_specs_in_catalog_order(
    registry: BackendRegistry,
) -> None:
    expected = tuple(
        spec
        for spec in registry.specs
        if isinstance(registry.states[spec.id], Available)
    )
    assert registry.available_specs() == expected


def test_registry_rejects_explicit_backend_that_conflicts_with_request(
    registry: BackendRegistry,
) -> None:
    request = BackendRequest(preferred_runtime=Runtime.CPP)
    with pytest.raises(BackendUnavailableError, match="no backend satisfies"):
        registry.create((0, 8), backend="py_boundary", request=request)

    narrow = replace(CATALOG_BY_ID["py_boundary"], coordinate_bits=32)
    narrow_registry = BackendRegistry(
        (narrow,), {narrow.id: Available(narrow.capabilities)}
    )
    with pytest.raises(BackendUnavailableError, match="no backend satisfies"):
        narrow_registry.create((0, 8), request=BackendRequest(coordinate_bits=64))
