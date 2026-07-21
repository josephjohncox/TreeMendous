"""Adversarial contracts for stable exact-batch scaling evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from scripts import verify_exact_batch_scaling as verifier
from tests.performance import exact_batch_scaling as scaling

CANDIDATE = "b" * 40


def _provenance(*, clean: bool = True) -> tuple[dict[str, Any], ...]:
    return (
        {"commit": CANDIDATE, "clean_worktree": clean},
        scaling._environment(),
        {
            "command": "python setup.py build_ext --inplace --force",
            "cxx": "c++ test",
            "cc": "cc test",
            "cflags": "-O3",
            "build_flags": {
                "TREE_MENDOUS_DISABLE_OPTIMIZATIONS": "",
                "TREE_MENDOUS_LOCAL_NATIVE": "",
                "TREE_MENDOUS_SANITIZERS": "",
                "TREE_MENDOUS_GLIBCXX_DEBUG": "",
            },
            "extensions": {
                "exact_batch": scaling._extension_metadata(
                    "treemendous.cpp._exact_batch"
                ),
                "cpp_boundary": scaling._extension_metadata("treemendous.cpp.boundary"),
            },
        },
    )


def _row(interval_count: int, *, value: int = 1_000_000) -> dict[str, Any]:
    samples = [value] * 20
    throughput = [scaling.BATCH_SIZE * 1_000_000_000 / item for item in samples]
    return {
        "interval_count": interval_count,
        "batch_size": scaling.BATCH_SIZE,
        "batch_latency_ns_samples": samples,
        "batch_latency_ns_median": value / 1,
        "batch_latency_ns_confidence_95": scaling._median_confidence(
            [item / 1 for item in samples]
        ),
        "logical_operations_per_second_median": 16_000.0,
        "logical_operations_per_second_confidence_95": scaling._median_confidence(
            throughput
        ),
        "packed_result_bytes": 408,
        "process_peak_rss_bytes": 100_000_000 + interval_count,
        "validated_sample_count": 20,
        "initial_and_final_interval_count": interval_count,
    }


def _report(monkeypatch: pytest.MonkeyPatch, *, clean: bool = True) -> dict[str, Any]:
    monkeypatch.setattr(scaling, "_provenance", lambda: _provenance(clean=clean))
    monkeypatch.setattr(scaling, "_peak_rss_bytes", lambda: 200_000_000)
    rows = {str(count): _row(count) for count in scaling.INTERVAL_COUNTS}
    return scaling._build_report(rows=rows, samples=20)


def _artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, clean: bool = True
) -> tuple[Path, dict[str, Any]]:
    report = _report(monkeypatch, clean=clean)
    path = tmp_path / "scaling.json"
    scaling.write_artifacts(report, path)
    monkeypatch.setattr(
        verifier,
        "_repository_state",
        lambda: {"commit": CANDIDATE, "clean_worktree": True},
    )
    return path, report


def _rewrite(path: Path, report: dict[str, Any]) -> None:
    scaling.write_artifacts(report, path)


def test_scaling_workload_is_restorative_and_canonically_attested() -> None:
    trace = scaling.trace_for_count(64)
    assert len(trace) == 16
    assert [row[0] for row in trace[:4]] == [
        scaling.MutationOpcode.DISCARD,
        scaling.MutationOpcode.DISCARD_REQUIRE_COVERED,
        scaling.MutationOpcode.ADD,
        scaling.MutationOpcode.ADD,
    ]
    before, results, packed_bytes = scaling._validate_case(
        64, trace, scaling._packed(trace)
    )
    assert len(before.intervals) == 64
    assert len(results) == 16
    assert packed_bytes == 408


def test_accepts_complete_scaling_triplet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _artifact(tmp_path, monkeypatch)
    result = verifier.verify_artifact(
        path,
        expected_candidate=CANDIDATE,
        require_samples=20,
        enforce_gate=True,
    )
    assert result["candidate_commit"] == CANDIDATE
    assert result["gate"]["passed"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["rows"]["100000"].update(
                batch_latency_ns_samples=[2_000_000] * 20
            ),
            "median is inconsistent",
        ),
        (
            lambda report: report["rows"]["100000"].update(
                batch_latency_ns_confidence_95=[1.0, 2.0]
            ),
            "CI is inconsistent",
        ),
        (
            lambda report: report["rows"]["100000"].update(
                logical_operations_per_second_median=1.0
            ),
            "throughput median is inconsistent",
        ),
    ],
)
def test_rejects_raw_sample_and_derived_value_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    message: str,
) -> None:
    path, report = _artifact(tmp_path, monkeypatch)
    mutation(report)
    _rewrite(path, report)
    with pytest.raises(ValueError, match=message):
        verifier.verify_artifact(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["workload_manifest"]["cases"]["64"].update(
                operations=[]
            ),
            "workload manifest",
        ),
        (
            lambda report: report["methodology"].update(interval_counts=[64, 1_000]),
            "methodology or matrix",
        ),
        (
            lambda report: report["rows"].pop("10000"),
            "matrix rows",
        ),
        (
            lambda report: report["resource_limits"].update(max_work_units=200_000_000),
            "resource limits",
        ),
        (
            lambda report: report["thresholds"].update(
                batch16_100000_upper_95_latency_ns=20_000_000
            ),
            "threshold",
        ),
        (
            lambda report: report["gates"]["production_envelope"].update(passed=False),
            "gate derivation",
        ),
        (
            lambda report: report["workload_manifest"]["cases"]["64"]["operations"][
                0
            ].__setitem__(0, True),
            "workload manifest",
        ),
        (
            lambda report: report["gates"]["production_envelope"].update(passed=1),
            "gate derivation",
        ),
        (
            lambda report: report["resource_limits"].update(
                max_live_intervals=100_000.0
            ),
            "resource limits",
        ),
        (
            lambda report: report["thresholds"].update(
                batch16_100000_upper_95_latency_ns=10_000_000.0
            ),
            "threshold",
        ),
        (
            lambda report: report["rows"]["100000"].update(interval_count=100_000.0),
            "declaration",
        ),
    ],
)
def test_rejects_workload_matrix_limit_threshold_and_gate_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    message: str,
) -> None:
    path, report = _artifact(tmp_path, monkeypatch)
    mutation(report)
    _rewrite(path, report)
    with pytest.raises(ValueError, match=message):
        verifier.verify_artifact(path)


def test_rejects_dirty_and_mismatched_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _artifact(tmp_path, monkeypatch, clean=False)
    with pytest.raises(ValueError, match="dirty worktree"):
        verifier.verify_artifact(path)

    path, _ = _artifact(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="required SHA"):
        verifier.verify_artifact(path, expected_candidate="e" * 40)

    monkeypatch.setattr(
        verifier,
        "_repository_state",
        lambda: {"commit": "e" * 40, "clean_worktree": True},
    )
    with pytest.raises(ValueError, match="verification checkout"):
        verifier.verify_artifact(path)


def test_rejects_binary_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, report = _artifact(tmp_path, monkeypatch)
    report["build"]["extensions"]["exact_batch"]["sha256"] = "0" * 64
    _rewrite(path, report)
    with pytest.raises(ValueError, match="loaded binary"):
        verifier.verify_artifact(path)


def test_rejects_missing_triplet_checksum_and_markdown_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _artifact(tmp_path, monkeypatch)
    path.with_suffix(".md").unlink()
    with pytest.raises(ValueError, match="missing"):
        verifier.verify_artifact(path)

    path, _ = _artifact(tmp_path, monkeypatch)
    Path(f"{path}.sha256").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verifier.verify_artifact(path)

    path, _ = _artifact(tmp_path, monkeypatch)
    path.with_suffix(".md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Markdown"):
        verifier.verify_artifact(path)


def test_exact_batch_workflow_publishes_three_sha_bound_triplets() -> None:
    repository = Path(__file__).parents[2]
    workflow = (repository / ".github/workflows/exact-batch-evidence.yml").read_text(
        encoding="utf-8"
    )
    assert "-m tests.performance.exact_batch_scaling" in workflow
    assert "-m scripts.verify_exact_batch_scaling" in workflow
    assert '--expected-candidate "$CANDIDATE_SHA"' in workflow
    assert 'test "$(find "$OUTPUT_DIR" -maxdepth 1 -type f | wc -l)" -eq 9' in workflow
    assert workflow.count("retention-days: 90") == 1
    for stem in ("exact-batch", "scalar-attribution", "exact-batch-scaling"):
        assert f"{stem}.json" in workflow
        assert f"{stem}.md" in workflow
        assert f"{stem}.json.sha256" in workflow
    assert "100,000 intervals batch-16 latency" in workflow


def test_native_and_sanitizer_lanes_run_full_stable_exact_suite() -> None:
    workflow = (
        Path(__file__).parents[2] / ".github" / "workflows" / "ci-cd.yaml"
    ).read_text(encoding="utf-8")
    native = workflow.split("  native-cpu:", 1)[1].split("  native-sanitizers:", 1)[0]
    sanitizers = workflow.split("  native-sanitizers:", 1)[1].split(
        "  benchmark-smoke:", 1
    )[0]
    assert "tests/unit/test_exact_batch.py" in native
    assert "tests/unit/test_exact_batch.py" in sanitizers
    assert "TSAN" not in sanitizers


def test_rejects_duplicate_keys_even_with_matching_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = _artifact(tmp_path, monkeypatch)
    encoded = (
        path.read_text(encoding="utf-8")
        .replace(
            f'"schema": "{scaling.SCHEMA}",',
            f'"schema": "{scaling.SCHEMA}",\n  "schema": "duplicate",',
            1,
        )
        .encode()
    )
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    Path(f"{path}.sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        verifier.verify_artifact(path)
