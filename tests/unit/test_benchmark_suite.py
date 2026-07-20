"""Contracts for reproducible benchmark profiles and durable artifacts."""

from __future__ import annotations

import hashlib
import json

import pytest

from tests.performance.application_workloads import (
    APPLICATION_SPECS,
    application_scenarios,
    qualify_application_scenarios,
)
from tests.performance.benchmark_suite import SCHEMA, write_artifacts
from tests.performance.payload_benchmark import qualify_payload_backends
from tests.performance.profiles import benchmark_profile

pytestmark = pytest.mark.unit


def test_smoke_profile_covers_canonical_geometry_operations_and_real_use_cases():
    profile = benchmark_profile("smoke")
    operations = {
        operation.kind
        for workload in profile.sampled_workloads + profile.qualification_workloads
        for operation in workload.setup + workload.operations
    }
    dimensions = {
        value
        for workload in profile.sampled_workloads + profile.qualification_workloads
        for key, value in workload.dimensions
        if key == "plausible_use"
    }

    assert profile.samples == 20
    assert {
        "add",
        "discard",
        "first_fit",
        "allocate",
        "cancel",
        "overlaps",
        "snapshot",
        "stats",
    } <= operations
    assert dimensions == {
        "allocator publishing state after each local mutation",
        "bounded compute-lane scheduling",
        "free-space allocator with release/reserve churn",
        "high-rate local free-space mutation",
        "read-mostly capacity and availability lookup",
        "sharded port, ID, address, or seat leasing",
    }


@pytest.mark.parametrize("profile_name", ("smoke", "standard", "large"))
def test_profiles_publish_mutation_fast_path_and_observation_costs(
    profile_name: str,
) -> None:
    names = {
        workload.name for workload in benchmark_profile(profile_name).sampled_workloads
    }
    assert {
        "canonical-local-mutation-throughput",
        "observed-fragmented-mutations",
    } <= names


def test_application_matrix_covers_fifty_distinct_interval_tasks():
    required = {
        "distributed-document-search",
        "distributed-regex-scan",
        "distributed-cluster-scheduling",
        "distributed-genetic-search",
        "genomic-annotation-overlap",
        "heap-free-space",
        "tcp-udp-port-leases",
    }
    ids = {spec.id for spec in APPLICATION_SPECS}
    scenarios = application_scenarios(scale=4, operations=20)

    assert len(APPLICATION_SPECS) == 50
    assert len(ids) == len(APPLICATION_SPECS)
    assert required <= ids
    assert {spec.category for spec in APPLICATION_SPECS} == {
        "distributed_partition",
        "scheduling_reservation",
        "overlap_catalog",
        "allocation_churn",
        "resource_leasing",
    }
    assert {spec.family for spec in APPLICATION_SPECS} == {
        "partition",
        "scheduling",
        "catalog",
        "allocator",
        "lease",
    }
    assert len(scenarios) == 50
    assert all(len(workload.operations) == 20 for _, workload in scenarios)
    operations_by_family: dict[str, set[str]] = {}
    for spec, workload in scenarios:
        operations_by_family.setdefault(spec.family, set()).update(
            operation.kind for operation in workload.operations
        )
    assert {"allocate", "cancel", "overlaps", "snapshot", "stats"} <= (
        operations_by_family["partition"]
    )
    assert {"overlaps", "first_fit", "snapshot", "stats"} <= (
        operations_by_family["catalog"]
    )
    assert {"add", "discard", "first_fit", "allocate"} <= (
        operations_by_family["allocator"]
    )


def test_all_application_scenarios_match_the_oracle():
    reports = qualify_application_scenarios(("py_boundary",), scale=2, operations=8)

    assert len(reports) == 50
    for report in reports:
        assert report["application"]["id"]
        assert len(report["validation"]["state_checksum"]) == 64
        assert report["results"]["py_boundary"]["execution_ns"] > 0


def test_every_payload_policy_is_qualified_with_real_operations():
    reports = qualify_payload_backends(("py_boundary",), scale=8, operations=35, seed=9)

    assert {report["workload"] for report in reports} == {
        "uniform-tenant-capacity",
        "joined-access-overlays",
        "ordered-booking-events",
    }
    for report in reports:
        assert report["validation"]["query_observations"] > 0
        assert len(report["validation"]["state_checksum"]) == 64
        assert report["results"]["py_boundary"]["execution_ns"] > 0


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown benchmark profile"):
        benchmark_profile("enormous")


def test_artifacts_are_atomic_human_readable_and_checksum_verified(tmp_path):
    report = {
        "schema": SCHEMA,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "profile": {"name": "test"},
        "backends": ["py_boundary"],
        "environment": {"commit": "abc123"},
        "sampled_reports": [],
        "qualification_reports": [],
        "application_reports": [],
        "payload_reports": [],
    }
    output = tmp_path / "run.json"

    json_path, markdown_path, checksum_path = write_artifacts(report, output)

    encoded = json_path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    try:
        parsed = json.loads(encoded)
    except json.JSONDecodeError as exc:
        pytest.fail(f"benchmark artifact is not valid JSON: {exc}")
    assert parsed["schema"] == SCHEMA
    assert digest in checksum_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "correctness-checked local measurements" in markdown
    assert digest in markdown
    assert not tuple(tmp_path.glob(".*.tmp"))
