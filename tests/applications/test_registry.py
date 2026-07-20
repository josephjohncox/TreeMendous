"""Contracts for the honest application implementation manifest."""

from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

import treemendous
from scripts import generate_scenario_catalog
from tests.performance import application_workloads
from treemendous.applications import (
    EXPECTED_FAMILY_COUNTS,
    SCENARIO_SPECS,
    SCENARIOS_BY_ID,
    ScenarioNotFoundError,
    ScenarioSpec,
    ScenarioStatus,
    create_application,
    get_scenario,
    list_scenarios,
    scenario_status_counts,
    validate_catalog_evidence,
    validate_completion_evidence,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_IDS = (
    "distributed-document-search",
    "distributed-regex-scan",
    "distributed-genetic-search",
    "distributed-graph-search",
    "distributed-sat-search",
    "distributed-fuzzing",
    "distributed-hyperparameter-search",
    "distributed-log-replay",
    "distributed-build-sharding",
    "map-reduce-input-splits",
    "distributed-web-crawl",
    "distributed-index-merge",
    "distributed-cluster-scheduling",
    "gpu-stream-scheduling",
    "render-farm-frames",
    "ci-runner-reservations",
    "meeting-room-booking",
    "airline-gate-assignment",
    "operating-room-booking",
    "laboratory-equipment-booking",
    "fleet-charging-windows",
    "radio-spectrum-timeslots",
    "warehouse-dock-appointments",
    "maintenance-window-planning",
    "genomic-annotation-overlap",
    "source-diagnostic-ranges",
    "filesystem-byte-locks",
    "database-key-range-locks",
    "packet-sequence-reassembly",
    "subtitle-cue-ranges",
    "video-edit-regions",
    "timeseries-alert-windows",
    "distributed-trace-spans",
    "morton-geospatial-ranges",
    "heap-free-space",
    "disk-block-allocation",
    "virtual-address-space",
    "database-page-allocation",
    "object-store-multipart-ranges",
    "cdn-byte-range-cache",
    "gpu-memory-arena",
    "ring-buffer-sequences",
    "tcp-udp-port-leases",
    "numeric-ip-address-pools",
    "database-id-pools",
    "software-license-seats",
    "warehouse-bin-ranges",
    "game-world-region-ids",
    "vlan-tag-pools",
    "phone-extension-pools",
)


def _complete_options() -> dict[str, Any]:
    return {
        "status": ScenarioStatus.COMPLETE,
        "engine": "builtins:dict",
        "example": "examples/applications/example.py",
        "oracle": "tests/oracles/applications/oracle.py",
        "benchmark": "tests/performance/applications/benchmark.py",
        "docs": "docs/scenarios/example.md",
    }


def _synthetic(**overrides: Any) -> ScenarioSpec:
    values: dict[str, Any] = {
        "id": "synthetic-scenario",
        "title": "Synthetic scenario",
        "category": "synthetic_category",
        "family": "partition",
        "description": "registry invariant fixture",
        **_complete_options(),
        **overrides,
    }
    return ScenarioSpec(**values)


def test_catalog_has_exact_ids_order_and_family_counts() -> None:
    assert tuple(spec.id for spec in SCENARIO_SPECS) == EXPECTED_IDS
    assert len(SCENARIO_SPECS) == len(set(EXPECTED_IDS)) == 50
    assert Counter(spec.family for spec in SCENARIO_SPECS) == Counter(
        EXPECTED_FAMILY_COUNTS
    )
    assert tuple(SCENARIOS_BY_ID) == EXPECTED_IDS


def test_all_scenarios_have_complete_validated_evidence() -> None:
    assert list_scenarios(status=ScenarioStatus.COMPLETE) == SCENARIO_SPECS
    assert not list_scenarios(status=ScenarioStatus.PLANNED)
    assert scenario_status_counts() == {
        ScenarioStatus.PLANNED: 0,
        ScenarioStatus.COMPLETE: 50,
    }
    validate_catalog_evidence(SCENARIO_SPECS, root=PROJECT_ROOT)
    for spec in SCENARIO_SPECS:
        assert spec.status is ScenarioStatus.COMPLETE
        for field in ("engine", "example", "oracle", "benchmark", "docs"):
            assert getattr(spec, field) is not None


def test_specs_and_registry_mappings_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        SCENARIO_SPECS[0].title = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        SCENARIOS_BY_ID["new-scenario"] = SCENARIO_SPECS[0]  # type: ignore[index]
    counts = scenario_status_counts()
    with pytest.raises(TypeError):
        counts[ScenarioStatus.COMPLETE] = 50  # type: ignore[index]


def test_lookup_filter_and_explicit_errors() -> None:
    assert get_scenario(EXPECTED_IDS[0]) is SCENARIO_SPECS[0]
    assert len(list_scenarios(family="catalog")) == 10
    with pytest.raises(ValueError, match="unknown scenario family"):
        list_scenarios(family="unknown")
    with pytest.raises(ScenarioNotFoundError):
        get_scenario("missing-scenario")
    application = create_application(EXPECTED_IDS[0])
    assert application.__class__.__module__.startswith(
        "treemendous.applications.partitioning"
    )
    with pytest.raises(ScenarioNotFoundError):
        create_application("missing-scenario")


def test_complete_status_requires_real_lazy_validated_evidence(tmp_path) -> None:
    complete = _synthetic()
    assert complete.status is ScenarioStatus.COMPLETE
    for field in ("engine", "example", "oracle", "benchmark", "docs"):
        with pytest.raises(ValueError, match="every artifact reference"):
            _synthetic(**{field: None})
    with pytest.raises(ValueError, match="module:callable"):
        _synthetic(engine="invalid-reference")

    # Construction is import-cycle safe: resolution is an explicit later gate.
    unresolved = _synthetic(engine="missing.scenario.module:create")
    with pytest.raises(ValueError, match="not resolvable"):
        validate_completion_evidence(unresolved, root=tmp_path)
    not_callable = _synthetic(engine="builtins:Ellipsis")
    with pytest.raises(ValueError, match="not callable"):
        validate_completion_evidence(not_callable, root=tmp_path)

    for reference in (
        complete.example,
        complete.oracle,
        complete.benchmark,
        complete.docs,
    ):
        assert reference is not None
        artifact = tmp_path / reference
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("evidence", encoding="utf-8")
    validate_completion_evidence(complete, root=tmp_path)
    validate_catalog_evidence((complete,), root=tmp_path)

    missing = tmp_path / complete.example
    missing.unlink()
    with pytest.raises(ValueError, match="artifact does not exist"):
        validate_completion_evidence(complete, root=tmp_path)
    with pytest.raises(ValueError, match="must match"):
        validate_completion_evidence(
            _synthetic(example="docs/not-an-example.py"),
            root=tmp_path,
        )


def test_planned_entries_cannot_expose_a_factory_or_use_invalid_ids() -> None:
    with pytest.raises(ValueError, match="planned scenarios cannot expose"):
        _synthetic(status=ScenarioStatus.PLANNED, engine="builtins:dict")
    with pytest.raises(ValueError, match="kebab-case"):
        _synthetic(id="Not_Kebab")
    with pytest.raises(ValueError, match="unknown scenario family"):
        _synthetic(family="unknown")
    with pytest.raises(TypeError, match="ScenarioStatus"):
        _synthetic(status="complete")


def test_benchmark_catalog_aliases_real_application_evidence() -> None:
    assert application_workloads.APPLICATION_SPECS is SCENARIO_SPECS
    assert application_workloads.ApplicationSpec is ScenarioSpec
    assert not hasattr(treemendous.applications, "_workload_for")
    for spec in application_workloads.APPLICATION_SPECS:
        assert spec.engine is not None
        module_name, separator, factory_name = spec.engine.partition(":")
        assert module_name.startswith("treemendous.applications.")
        assert separator == ":"
        assert factory_name.startswith("create_")


def test_generator_check_is_side_effect_free_on_drift(tmp_path, monkeypatch) -> None:
    document = tmp_path / "use-cases.md"
    stale = generate_scenario_catalog.render_status_block().replace(
        "`COMPLETE`", "`DRIFTED`", 1
    )
    document.write_text(stale, encoding="utf-8")
    monkeypatch.setattr(generate_scenario_catalog, "DOCUMENT", document)

    assert generate_scenario_catalog.main(["--check"]) == 1
    assert document.read_text(encoding="utf-8") == stale
    assert generate_scenario_catalog.main([]) == 0
    assert generate_scenario_catalog.main(["--check"]) == 0


def test_generator_rejects_duplicate_status_blocks(tmp_path, monkeypatch) -> None:
    document = tmp_path / "use-cases.md"
    block = generate_scenario_catalog.render_status_block()
    document.write_text(f"{block}\n{block}", encoding="utf-8")
    monkeypatch.setattr(generate_scenario_catalog, "DOCUMENT", document)

    assert generate_scenario_catalog.main(["--check"]) == 2


def test_registry_remains_outside_the_stable_root_api() -> None:
    assert "ScenarioSpec" not in treemendous.__all__
    assert "create_application" not in treemendous.__all__
    assert not hasattr(treemendous, "ScenarioSpec")
    assert not hasattr(treemendous, "create_application")
