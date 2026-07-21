"""Adversarial contracts for durable experimental exact-batch evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from scripts import verify_exact_batch_benchmark as verifier
from tests.performance import exact_batch_benchmark as benchmark
from tests.performance.mutation_attribution import paired_statistics

CANDIDATE = "b" * 40
BASELINE = "a" * 40


def _provenance(*, clean: bool = True) -> tuple[dict[str, Any], ...]:
    return (
        {"commit": CANDIDATE, "clean_worktree": clean},
        {
            "python": "3.12.7",
            "implementation": "CPython",
            "python_compiler": "GCC 13",
            "platform": "Linux-test",
            "machine": "x86_64",
            "processor": "test-cpu",
            "cpu_count": 4,
        },
        {
            "command": "python setup.py build_ext --inplace --force",
            "cxx": "c++ 13",
            "cc": "gcc",
            "cflags": "-O3",
            "build_flags": {
                "TREE_MENDOUS_DISABLE_OPTIMIZATIONS": "",
                "TREE_MENDOUS_LOCAL_NATIVE": "",
                "TREE_MENDOUS_SANITIZERS": "",
                "TREE_MENDOUS_GLIBCXX_DEBUG": "",
            },
            "extensions": {
                "exact_batch": benchmark._extension_metadata(
                    "treemendous.cpp._exact_batch"
                ),
                "cpp_boundary": benchmark._extension_metadata(
                    "treemendous.cpp.boundary"
                ),
            },
        },
    )


def _report(monkeypatch: pytest.MonkeyPatch, *, clean: bool = True) -> dict[str, Any]:
    monkeypatch.setattr(benchmark, "_provenance", lambda: _provenance(clean=clean))
    rows: dict[str, Any] = {}
    for size in benchmark.BATCH_SIZES:
        batch_value = 100 if size == 16 else 500
        scalar_value = 1_000
        batch = [batch_value] * 20
        scalar = [scalar_value] * 20
        comparison = paired_statistics(scalar, batch, ratio_limit=1.0, seed=50)
        throughput = [1_000_000_000 / value for value in batch]
        rows[str(size)] = {
            "batch_size": size,
            "classification": (
                "single-call-no-op-diagnostic" if size == 1 else "restorative"
            ),
            "baseline_backend": "cpp_boundary",
            "logical_operations_per_sample": size * 20,
            "batch_ns_per_operation_samples": batch,
            "scalar_ns_per_operation_samples": scalar,
            "batch_median_ops_per_second": 1_000_000_000 / batch_value,
            "batch_ops_per_second_confidence_95": benchmark._median_confidence(
                throughput
            ),
            "paired": comparison,
            "speedup_confidence_95": (
                1.0 / comparison["confidence_95_ratio"][1],
                1.0 / comparison["confidence_95_ratio"][0],
            ),
        }
    return benchmark._build_report(
        rows=rows,
        materialize_samples=[50] * 20,
        samples=20,
        target_operations=20,
    )


def _artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, clean: bool = True
) -> tuple[Path, dict[str, Any]]:
    report = _report(monkeypatch, clean=clean)
    path = tmp_path / "exact.json"
    benchmark.write_artifacts(report, path)
    return path, report


def _rewrite(path: Path, report: dict[str, Any]) -> None:
    benchmark.write_artifacts(report, path)


def test_exact_artifact_triplet_and_all_fixed_derivations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _artifact(tmp_path, monkeypatch)
    result = verifier.verify_artifact(
        path, expected_candidate=CANDIDATE, require_samples=20
    )
    assert result["candidate_commit"] == CANDIDATE
    assert result["gates"]["batch16_absolute"]["passed"]
    assert result["gates"]["batch16_speedup"]["passed"]
    assert result["gates"]["break_even_by_4"]["passed"]


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (None, "exactly 20 positive"),
        ([100] * 19, "exactly 20 positive"),
        ([100.0] * 20, "exactly 20 positive"),
    ],
)
def test_rejects_malformed_missing_and_mistyped_raw_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: object,
    message: str,
) -> None:
    path, report = _artifact(tmp_path, monkeypatch)
    report["rows"]["16"]["batch_ns_per_operation_samples"] = replacement
    _rewrite(path, report)
    with pytest.raises(ValueError, match=message):
        verifier.verify_artifact(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report["gates"]["batch16_speedup"].update(passed=False),
        lambda report: report["rows"]["16"].update(speedup_confidence_95=[99.0, 100.0]),
        lambda report: report["rows"]["16"]["paired"].update(median_ratio=0.01),
    ],
)
def test_rejects_falsified_gates_and_derived_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
) -> None:
    path, report = _artifact(tmp_path, monkeypatch)
    mutation(report)
    _rewrite(path, report)
    with pytest.raises(ValueError, match="inconsistent"):
        verifier.verify_artifact(path)


def test_rejects_py_boundary_substitution_and_workload_digest_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, report = _artifact(tmp_path, monkeypatch)
    report["baseline_backend"] = "py_boundary"
    _rewrite(path, report)
    with pytest.raises(ValueError, match="cpp_boundary"):
        verifier.verify_artifact(path)

    path, report = _artifact(tmp_path, monkeypatch)
    report["workload_manifest"]["digest"] = "0" * 64
    _rewrite(path, report)
    with pytest.raises(ValueError, match="workload manifest"):
        verifier.verify_artifact(path)


@pytest.mark.parametrize("extension_name", ["exact_batch", "cpp_boundary"])
@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("path", "path does not match loaded module"),
        ("sha256", "digest does not match loaded module"),
    ],
)
def test_rejects_extension_path_and_digest_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extension_name: str,
    field: str,
    message: str,
) -> None:
    path, report = _artifact(tmp_path, monkeypatch)
    extensions = report["build"]["extensions"]
    if field == "path":
        other_name = (
            "cpp_boundary" if extension_name == "exact_batch" else "exact_batch"
        )
        replacement = extensions[other_name]["path"]
    else:
        replacement = "0" * 64
    extensions[extension_name][field] = replacement
    _rewrite(path, report)
    with pytest.raises(ValueError, match=message):
        verifier.verify_artifact(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report["methodology"].update(target_operations=21),
        lambda report: report["rows"]["16"].update(
            logical_operations_per_sample=16 * 21
        ),
    ],
)
def test_rejects_target_operations_and_logical_count_inconsistency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
) -> None:
    path, report = _artifact(tmp_path, monkeypatch)
    mutation(report)
    _rewrite(path, report)
    with pytest.raises(ValueError, match="declaration is invalid"):
        verifier.verify_artifact(path)


def test_rejects_commit_mismatch_and_dirty_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _artifact(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="commit does not match"):
        verifier.verify_artifact(path, expected_candidate="e" * 40)

    path, _ = _artifact(tmp_path, monkeypatch, clean=False)
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


def _quick_scalar_report(
    *, candidate_ns: int = 10_000, candidate: str = CANDIDATE
) -> dict[str, Any]:
    rounds = []
    for _ in range(20):
        rounds.append(
            {
                "baseline": {
                    "environment": {"commit": BASELINE, "dirty": "false"},
                    "timings_ns": {"layers": {"rangeset_public": 10_000}},
                },
                "candidate": {
                    "environment": {"commit": candidate, "dirty": "false"},
                    "timings_ns": {"layers": {"rangeset_public": candidate_ns}},
                },
            }
        )
    return {
        "methodology": {"full_representative_suite": False},
        "baseline": {"commit": BASELINE, "dirty": "false"},
        "candidate": {"commit": candidate, "dirty": "false"},
        "round_evidence": rounds,
        "semantic_checksums_match": True,
        "environments_match": True,
        "binary_provenance_complete": True,
        "clean_worktrees": True,
        "status": "diagnostic",
        "promotion_eligible": False,
    }


def test_complete_gate_rejects_absent_scalar_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _artifact(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="requires scalar attribution"):
        verifier.verify_artifact(path, enforce_gates=True)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("semantic_checksums_match", False, "semantic evidence"),
        ("environments_match", False, "environments do not match"),
        ("binary_provenance_complete", False, "provenance is incomplete"),
        ("clean_worktrees", False, "worktrees are not clean"),
        ("status", "invalid", "validity status"),
        ("promotion_eligible", True, "validity status"),
    ],
)
def test_quick_scalar_gate_rejects_invalid_integrity_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
    message: str,
) -> None:
    scalar = tmp_path / "scalar.json"
    scalar.touch()
    report = _quick_scalar_report()
    report[field] = replacement
    monkeypatch.setattr(verifier, "verify_attribution", lambda *args, **kwargs: report)
    with pytest.raises(ValueError, match=message):
        verifier.verify_scalar_attribution(scalar, expected_candidate=CANDIDATE)


def test_quick_scalar_gate_recomputes_ci_and_binds_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scalar = tmp_path / "scalar.json"
    scalar.touch()
    monkeypatch.setattr(
        verifier, "verify_attribution", lambda *args, **kwargs: _quick_scalar_report()
    )
    result = verifier.verify_scalar_attribution(
        scalar, expected_candidate=CANDIDATE, expected_baseline=BASELINE
    )
    assert result["rangeset_public_confidence_95_ratio"][1] == 1.0

    monkeypatch.setattr(
        verifier,
        "verify_attribution",
        lambda *args, **kwargs: _quick_scalar_report(candidate_ns=10_400),
    )
    with pytest.raises(ValueError, match="exceeds 1.03"):
        verifier.verify_scalar_attribution(scalar, expected_candidate=CANDIDATE)

    monkeypatch.setattr(
        verifier,
        "verify_attribution",
        lambda *args, **kwargs: _quick_scalar_report(candidate="e" * 40),
    )
    with pytest.raises(ValueError, match="does not match exact-batch candidate"):
        verifier.verify_scalar_attribution(scalar, expected_candidate=CANDIDATE)


def test_exact_batch_workflow_uses_package_module_entry_points() -> None:
    workflow = (
        Path(__file__).parents[2] / ".github" / "workflows" / "exact-batch-evidence.yml"
    ).read_text(encoding="utf-8")

    assert workflow.startswith("name: Exact-batch production evidence\n")
    assert "Verified stable exact-batch evidence" in workflow
    assert "-m tests.performance.exact_batch_benchmark" in workflow
    assert "-m scripts.verify_mutation_attribution" in workflow
    assert "-m scripts.verify_exact_batch_benchmark" in workflow
    assert (
        "python \\\n            tests/performance/exact_batch_benchmark.py"
        not in workflow
    )
    assert "python \\\n            scripts/verify_" not in workflow
