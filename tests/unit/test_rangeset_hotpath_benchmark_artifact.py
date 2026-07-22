"""Adversarial contracts for the range-set hot-path benchmark artifact."""

from __future__ import annotations

import hashlib
import statistics
from pathlib import Path
from typing import Any

import pytest

from scripts import verify_rangeset_hotpath_benchmark as verifier
from tests.performance import rangeset_hotpath_benchmark as benchmark

CANDIDATE = "b" * 40


def _provenance() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        {"commit": CANDIDATE, "clean_worktree": True},
        {
            "python": "3.12.7",
            "implementation": "CPython",
            "python_compiler": "Clang 17",
            "platform": "macOS-test",
            "machine": "arm64",
            "processor": "test-cpu",
            "cpu_count": 12,
        },
        {
            "command": "python setup.py build_ext --inplace --force",
            "cxx": "c++ 17",
            "cc": "clang",
            "cflags": "-O3",
            "build_flags": {
                "TREE_MENDOUS_DISABLE_OPTIMIZATIONS": "",
                "TREE_MENDOUS_LOCAL_NATIVE": "",
                "TREE_MENDOUS_SANITIZERS": "",
                "TREE_MENDOUS_GLIBCXX_DEBUG": "",
            },
            "extensions": {
                "cpp_boundary": benchmark._extension_metadata(
                    "treemendous.cpp.boundary"
                )
            },
        },
    )


def _report(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(benchmark, "_provenance", _provenance)
    samples = 20
    target_operations = 20 * len(benchmark._TRACE)
    paths: dict[str, Any] = {}
    operations_per_trace = len(benchmark._TRACE)
    for index, name in enumerate(benchmark.PATHS):
        ns_samples = [100 + index * 10 + (sample % 3) for sample in range(samples)]
        throughput = [1_000_000_000 / value for value in ns_samples]
        paths[name] = {
            "interface": benchmark._PATH_RUNNERS[name][3],
            "operations_per_trace": operations_per_trace,
            "logical_operations_per_sample": (
                max(20, target_operations // operations_per_trace)
                * operations_per_trace
            ),
            "ns_per_operation_samples": ns_samples,
            "median_ns_per_operation": statistics.median(ns_samples),
            "median_ops_per_second": statistics.median(throughput),
            "ops_per_second_confidence_95": list(
                benchmark._median_confidence(throughput)
            ),
        }
    candidate, environment, build = benchmark._provenance()
    return {
        "schema": benchmark.SCHEMA,
        "candidate": candidate,
        "environment": environment,
        "build": build,
        "backend": benchmark.BACKEND,
        "methodology": {
            "samples": samples,
            "target_operations": target_operations,
            "workload": "deterministic restorative trace",
            "timed_layer": "public mutation calls only",
            "excluded": "construction, setup, validation, snapshots",
            "throughput_bootstrap_seed": benchmark.THROUGHPUT_BOOTSTRAP_SEED,
            "bootstrap_resamples": benchmark.BOOTSTRAP_RESAMPLES,
            "paths": list(benchmark.PATHS),
            "universal_claim": False,
        },
        "workload_manifest": benchmark.workload_manifest(),
        "paths": paths,
    }


def _artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any]]:
    report = _report(monkeypatch)
    path = tmp_path / "hotpath.json"
    benchmark.write_artifacts(report, path)
    return path, report


def test_artifact_triplet_verifies_and_recomputes_all_derivations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _artifact(tmp_path, monkeypatch)
    result = verifier.verify_artifact(
        path, expected_candidate=CANDIDATE, require_samples=20
    )
    assert result["candidate_commit"] == CANDIDATE
    assert result["samples"] == 20
    assert result["workload_digest"] == benchmark.workload_manifest()["digest"]


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (None, "exactly 20 positive"),
        ([100] * 19, "exactly 20 positive"),
        ([100.0] * 20, "exactly 20 positive"),
        ([0] * 20, "exactly 20 positive"),
    ],
)
def test_rejects_malformed_sample_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: object,
    message: str,
) -> None:
    path, report = _artifact(tmp_path, monkeypatch)
    report["paths"]["scalar_synchronized"]["ns_per_operation_samples"] = replacement
    benchmark.write_artifacts(report, path)
    with pytest.raises(ValueError, match=message):
        verifier.verify_artifact(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report["paths"]["native_floor"].update(
            median_ops_per_second=1.0
        ),
        lambda report: report["paths"]["native_floor"].update(
            ops_per_second_confidence_95=[1.0, 2.0]
        ),
        lambda report: report["paths"]["scalar_synchronized"].update(
            median_ns_per_operation=1
        ),
    ],
)
def test_rejects_falsified_derived_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: Any
) -> None:
    path, report = _artifact(tmp_path, monkeypatch)
    mutation(report)
    benchmark.write_artifacts(report, path)
    with pytest.raises(ValueError, match="inconsistent"):
        verifier.verify_artifact(path)


def test_rejects_universal_claim_and_backend_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, report = _artifact(tmp_path, monkeypatch)
    report["methodology"]["universal_claim"] = True
    benchmark.write_artifacts(report, path)
    with pytest.raises(ValueError, match="methodology is inconsistent"):
        verifier.verify_artifact(path)

    path, report = _artifact(tmp_path, monkeypatch)
    report["backend"] = "py_boundary"
    benchmark.write_artifacts(report, path)
    with pytest.raises(ValueError, match="cpp_boundary"):
        verifier.verify_artifact(path)


def test_rejects_workload_and_extension_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, report = _artifact(tmp_path, monkeypatch)
    report["workload_manifest"]["digest"] = "0" * 64
    benchmark.write_artifacts(report, path)
    with pytest.raises(ValueError, match="workload manifest"):
        verifier.verify_artifact(path)

    path, report = _artifact(tmp_path, monkeypatch)
    report["build"]["extensions"]["cpp_boundary"]["sha256"] = "0" * 64
    benchmark.write_artifacts(report, path)
    with pytest.raises(ValueError, match="digest does not match loaded module"):
        verifier.verify_artifact(path)


def test_rejects_commit_mismatch_and_dirty_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _artifact(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="commit does not match"):
        verifier.verify_artifact(path, expected_candidate="e" * 40)

    path, report = _artifact(tmp_path, monkeypatch)
    report["candidate"]["clean_worktree"] = False
    benchmark.write_artifacts(report, path)
    with pytest.raises(ValueError, match="dirty worktree"):
        verifier.verify_artifact(path)


def test_rejects_checksum_and_markdown_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _artifact(tmp_path, monkeypatch)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        verifier.verify_artifact(path)

    path, _ = _artifact(tmp_path, monkeypatch)
    path.with_suffix(".md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Markdown"):
        verifier.verify_artifact(path)


def test_rejects_duplicate_json_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _artifact(tmp_path, monkeypatch)
    encoded = (
        path.read_text(encoding="utf-8")
        .replace(
            f'"schema": "{benchmark.SCHEMA}",',
            f'"schema": "{benchmark.SCHEMA}",\n  "schema": "duplicate",',
            1,
        )
        .encode()
    )
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    Path(f"{path}.sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        verifier.verify_artifact(path)
