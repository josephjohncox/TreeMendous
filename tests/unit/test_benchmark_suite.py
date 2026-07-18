"""Contracts for reproducible benchmark profiles and durable artifacts."""

from __future__ import annotations

import hashlib
import json

import pytest

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
    assert len(dimensions) == 4


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
