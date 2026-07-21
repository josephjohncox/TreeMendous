"""Durable benchmark artifact and ad hoc workflow contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from scripts.verify_benchmark_artifact import STABLE_BACKENDS, verify_artifact
from tests.performance.benchmark_suite import SCHEMA, write_artifacts
from tests.performance.harness import timing_statistics


def _timing(samples: list[int]) -> dict[str, object]:
    return {**asdict(timing_statistics(samples)), "samples_ns": samples}


def _sampled_report(workload: str) -> dict[str, object]:
    samples = list(range(100, 120))
    return {
        "workload": workload,
        "dataset": {"actual_interval_count": 64, "timed_operations": 1_000},
        "methodology": {"independent_runs": len(samples)},
        "results": {
            "py_boundary": {
                "execution": _timing(samples),
                "setup": _timing([value + 20 for value in samples]),
                "validation_overhead": _timing([value + 40 for value in samples]),
                "operation_latency": {"add": {"per_run_median": _timing(samples)}},
            }
        },
    }


def _report() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "generated_at": "2026-01-01T00:00:00+00:00",
        "profile": {"name": "standard", "section": "sampled"},
        "backends": list(STABLE_BACKENDS),
        "environment": {"commit": "a" * 40},
        "ci_provenance": {"github_sha": "a" * 40},
        "sampled_reports": [
            _sampled_report("canonical-local-mutation-throughput"),
            _sampled_report("observed-fragmented-mutations"),
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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda timing: timing.pop("samples_ns"), "samples_ns must be an array"),
        (lambda timing: timing.__setitem__("samples_ns", [100] * 19), "sample count"),
        (
            lambda timing: timing.__setitem__("samples_ns", [100.0] * 20),
            "positive signed 64-bit integers",
        ),
        (lambda timing: timing.__setitem__("median_ns", 1), "inconsistent median_ns"),
    ],
)
def test_schema_v4_verifier_rejects_malformed_or_inconsistent_samples(
    tmp_path: Path, mutation: object, message: str
) -> None:
    report = _report()
    sampled = report["sampled_reports"]
    assert isinstance(sampled, list)
    first = sampled[0]
    assert isinstance(first, dict)
    results = first["results"]
    assert isinstance(results, dict)
    backend = results["py_boundary"]
    assert isinstance(backend, dict)
    execution = backend["execution"]
    assert isinstance(execution, dict)
    assert callable(mutation)
    mutation(execution)
    output = tmp_path / "malformed.json"
    write_artifacts(report, output)

    with pytest.raises(ValueError, match=message):
        verify_artifact(output)


def test_schema_v4_verifier_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    output = tmp_path / "duplicate.json"
    write_artifacts(_report(), output)
    encoded = (
        output.read_text(encoding="utf-8")
        .replace(
            '"schema": "treemendous-validated-benchmark-suite-v4"',
            '"schema": "treemendous-validated-benchmark-suite-v4",\n  "schema": "duplicate"',
            1,
        )
        .encode()
    )
    output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    Path(f"{output}.sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    markdown = output.with_suffix(".md")
    markdown.write_text(
        markdown.read_text(encoding="utf-8").replace(
            "JSON SHA-256: `", f"JSON SHA-256: `{digest}`\nold digest: `", 1
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid"):
        verify_artifact(output)


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
