from __future__ import annotations

import pytest

import treemendous.backend_manager as facade
from treemendous.backends import Available, Capability, Invalid, probe_backend
from treemendous.backends.types import BackendSpec, ProbeState
from treemendous.basic.protocols import ImplementationType
from treemendous.domain import BackendInvalidError, BackendUnavailableError, Span


@pytest.fixture(scope="module")
def manager() -> facade.TreeMendousBackendManager:
    return facade.TreeMendousBackendManager()


def _manager_with_state(
    backend_id: str, state: ProbeState
) -> facade.TreeMendousBackendManager:
    def probe(spec: BackendSpec) -> ProbeState:
        return state if spec.id == backend_id else probe_backend(spec)

    return facade.TreeMendousBackendManager(probe=probe)


def test_available_backend_filters_are_consistent_and_case_insensitive(
    manager: facade.TreeMendousBackendManager,
) -> None:
    available = manager.get_available_backends()
    assert available
    assert all(info.config.available for info in available.values())

    python = manager.get_backends_by_language("PyThOn")
    boundary = manager.get_backends_by_type(ImplementationType.BOUNDARY)
    analytics = manager.get_backends_with_feature("analytics")

    assert python
    assert all(info.config.language == "python" for info in python.values())
    assert boundary
    assert all(
        info.config.implementation_type is ImplementationType.BOUNDARY
        for info in boundary.values()
    )
    assert set(analytics) == {"py_summary", "py_boundary_summary"}
    assert not manager.get_backends_by_language("fortran")
    assert not manager.get_backends_with_feature("imaginary-feature")
    assert manager.select_best_backend() in available


def test_legacy_manager_optional_features_and_metadata(
    manager: facade.TreeMendousBackendManager,
) -> None:
    basic = manager.create_manager("py_boundary")
    basic.release_interval(0, 10)
    basic.reserve_interval(3, 5, data="ignored")

    observed = [item.span for item in basic.get_intervals()]
    assert observed == [Span(0, 3), Span(5, 10)]
    assert basic.get_availability_stats() is None
    assert basic.find_best_fit(1) is None
    assert basic.find_largest_available() is None
    assert basic.sample_random_interval() is None
    assert basic.verify_properties()
    assert basic.get_performance_stats().operation_count == 2
    assert basic.get_backend_info().config.implementation_id == "py_boundary"
    assert basic.get_implementation_type() is ImplementationType.BOUNDARY
    assert (
        basic.get_performance_tier() is basic.get_backend_info().config.performance_tier
    )
    assert basic.supports_feature("core-operations")
    assert not basic.supports_feature("random-sampling")

    summary = manager.create_manager("py_summary")
    summary.release_interval(0, 20)
    summary.reserve_interval(4, 7)
    stats = summary.get_availability_stats()
    best = summary.find_best_fit(3)
    largest = summary.find_largest_available()
    assert stats is not None and stats.total_free == 17
    assert best is not None and best.length == 3
    assert largest is not None and largest.span == Span(7, 20)
    assert summary.get_performance_stats().operation_count >= 2

    treap = manager.create_manager("py_treap", random_seed=17)
    treap.release_interval(10, 15)
    sampled = treap.sample_random_interval()
    assert sampled is not None and sampled.span == Span(10, 15)
    assert treap.verify_properties()


def test_explicit_resolution_uses_validated_not_declared_capabilities() -> None:
    manager = _manager_with_state(
        "py_boundary", Available(frozenset({Capability.CORE}))
    )
    with pytest.raises(BackendUnavailableError, match="lacks required capabilities"):
        manager.create_range_set(
            (0, 10),
            "py_boundary",
            require=frozenset({Capability.CORE, Capability.PAYLOADS}),
        )


def test_explicit_resolution_distinguishes_unknown_invalid_and_unavailable(
    manager: facade.TreeMendousBackendManager,
) -> None:
    with pytest.raises(BackendUnavailableError, match="unknown backend"):
        manager.create_manager("missing")
    with pytest.raises(BackendUnavailableError, match="experimental"):
        manager.create_manager("gpu_boundary_summary")

    invalid_manager = _manager_with_state("py_boundary", Invalid("broken law"))
    with pytest.raises(BackendInvalidError, match="broken law"):
        invalid_manager.create_manager("py_boundary")


def test_constructor_options_and_module_level_facade(
    manager: facade.TreeMendousBackendManager,
) -> None:
    ranges = manager.create_range_set(
        (0, 8),
        "py_boundary",
        initially_available=False,
    )
    assert not ranges.intervals()
    ranges.add(Span(2, 6))
    assert ranges.snapshot().total_free == 4

    with pytest.raises(TypeError):
        manager.create_manager("py_boundary", not_a_constructor_option=True)

    assert facade.get_backend_manager() is facade.get_backend_manager()
    assert facade.list_available_backends()
    created = facade.create_interval_tree("py_boundary")
    created.release_interval(0, 2)
    assert created.get_total_available_length() == 2
    public_ranges = facade.create_range_set(
        (0, 2), "py_boundary", initially_available=False
    )
    assert not public_ranges.intervals()


def test_print_backend_status_covers_the_catalog(
    manager: facade.TreeMendousBackendManager, capsys: pytest.CaptureFixture[str]
) -> None:
    manager.print_backend_status()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(facade.CATALOG)
    assert all(
        spec.id in line for spec, line in zip(facade.CATALOG, lines, strict=True)
    )
