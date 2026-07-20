"""Durable benchmark artifact and ad hoc workflow contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_benchmark_artifact import STABLE_BACKENDS, verify_artifact
from tests.performance.benchmark_suite import SCHEMA, write_artifacts


def _report() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "profile": {"name": "standard", "section": "sampled"},
        "backends": list(STABLE_BACKENDS),
        "environment": {"commit": "a" * 40},
        "ci_provenance": {"github_sha": "a" * 40},
        "sampled_reports": [
            {
                "workload": "canonical-local-mutation-throughput",
                "results": {},
            },
            {"workload": "observed-fragmented-mutations", "results": {}},
        ],
        "qualification_reports": [],
        "application_reports": [],
        "payload_reports": [],
    }


def test_benchmark_verifier_accepts_complete_canonical_bundle(tmp_path: Path) -> None:
    output = tmp_path / "standard-sampled.json"
    write_artifacts(_report(), output)

    verified = verify_artifact(
        output,
        expected_profile="standard",
        expected_section="sampled",
        expected_commit="a" * 40,
        required_workloads=(
            "canonical-local-mutation-throughput",
            "observed-fragmented-mutations",
        ),
        require_all_stable=True,
    )

    expected_workloads = (
        "canonical-local-mutation-throughput",
        "observed-fragmented-mutations",
    )
    assert verified.json_path == output
    assert len(verified.digest) == 64
    assert verified.workloads == expected_workloads


def test_benchmark_verifier_rejects_tampering_and_missing_workload(
    tmp_path: Path,
) -> None:
    output = tmp_path / "standard-sampled.json"
    write_artifacts(_report(), output)
    output.write_text(json.dumps(_report()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum sidecar"):
        verify_artifact(output)

    write_artifacts(_report(), output)
    with pytest.raises(ValueError, match="required sampled workloads"):
        verify_artifact(output, required_workloads=("missing-workload",))


def test_github_workflows_publish_verified_benchmark_triplets() -> None:
    root = Path(__file__).parents[2]
    ad_hoc = (root / ".github/workflows/benchmarks-adhoc.yml").read_text(
        encoding="utf-8"
    )
    scheduled = (root / ".github/workflows/benchmarks.yml").read_text(encoding="utf-8")
    pull_request = (root / ".github/workflows/ci-cd.yaml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in ad_hoc
    assert "GITHUB_STEP_SUMMARY" in ad_hoc
    for workflow in (ad_hoc, scheduled, pull_request):
        assert "canonical-local-mutation-throughput" in workflow
        assert "observed-fragmented-mutations" in workflow
        assert "scripts/verify_benchmark_artifact.py" in workflow
        assert "--require-all-stable" in workflow
        assert (
            "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
        ) in workflow
    assert "retention-days: 90" in ad_hoc
    assert "retention-days: 90" in scheduled
