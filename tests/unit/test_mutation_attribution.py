"""Contracts for paired native-mutation attribution and explicit promotion gates."""

from __future__ import annotations

import copy
import hashlib
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.verify_mutation_attribution import verify_artifact, verify_report
from tests.performance import mutation_attribution as attribution
from tests.performance.mutation_attribution import (
    FOCUSED_WORKLOAD_NAMES,
    PRIMARY_LAYER,
    PYTHON_CONTROLS,
    REQUIRED_LAYERS,
    SCHEMA,
    paired_statistics,
    representative_workload_manifest,
    write_artifacts,
)

PRIMARY_LIMIT = 0.85
REGRESSION_LIMIT = 1.05
CONTROL_BOUNDS = (0.95, 1.05)


def _samples(value: int) -> list[int]:
    return [value] * 20


def _environment(commit: str, *, dirty: bool = False) -> dict[str, str]:
    return {
        "commit": commit,
        "dirty": str(dirty).lower(),
        "python": "3.12.7",
        "build_command": "${PYTHON} setup.py build_ext --inplace --force --verbose",
        "compiler_invocations": "c++ -c boundary.cpp",
        "build_flags": "{}",
        "extension_path": "treemendous/cpp/boundary.so",
        "extension_sha256": "e" * 64,
    }


def _passing_report(*, dirty: bool = False) -> dict[str, Any]:
    manifest = representative_workload_manifest()
    representative_names = tuple(item["name"] for item in manifest["workloads"])
    primary = paired_statistics(_samples(100), _samples(80), ratio_limit=PRIMARY_LIMIT)
    regression = paired_statistics(
        _samples(100), _samples(100), ratio_limit=REGRESSION_LIMIT
    )
    section_names = {
        "layers": REQUIRED_LAYERS,
        "focused": FOCUSED_WORKLOAD_NAMES,
        "representative": representative_names,
        "controls": PYTHON_CONTROLS,
    }
    semantics: dict[str, dict[str, dict[str, Any]]] = {}
    for section, names in section_names.items():
        entries: dict[str, dict[str, Any]] = {}
        for name in names:
            counters = {
                "operation_count": 1_000,
                "no_op_count": 0,
                "touched_interval_count": 1_000,
                "touched_length": 1_000,
            }
            if section == "layers":
                entries[name] = {
                    "evidence_checksum": "d" * 64,
                    "state_checksum": "c" * 64,
                    **counters,
                }
            else:
                entries[name] = {
                    **({"trace_digest": "f" * 64} if section != "controls" else {}),
                    "state_checksum": "c" * 64,
                    "query_checksum": "b" * 64,
                    "successful_operation_count": 1_000,
                    **counters,
                }
        semantics[section] = entries
    baseline_environment = _environment("a" * 40, dirty=dirty)
    candidate_environment = _environment("b" * 40, dirty=dirty)
    rounds = []
    for index in range(20):
        entry: dict[str, Any] = {
            "round": index,
            "execution_order": (
                ["baseline", "candidate"]
                if index % 2 == 0
                else ["candidate", "baseline"]
            ),
        }
        for label, environment in (
            ("baseline", baseline_environment),
            ("candidate", candidate_environment),
        ):
            entry[label] = {
                "environment": environment.copy(),
                "trace_digest": attribution._checksum(manifest["workloads"][0]),
                "representative_manifest_digest": manifest["digest"],
                "timings_ns": {
                    section: {
                        name: (
                            80
                            if label == "candidate"
                            and section == "layers"
                            and name == PRIMARY_LAYER
                            else 100
                        )
                        for name in names
                    }
                    for section, names in section_names.items()
                },
                "semantic": semantics,
            }
        rounds.append(entry)
    semantic_evidence = {
        section: {
            name: {"baseline": value, "candidate": value}
            for name, value in entries.items()
        }
        for section, entries in semantics.items()
    }
    status = "diagnostic" if dirty else "pass"
    return {
        "schema": SCHEMA,
        "methodology": {
            "samples": 20,
            "primary_ratio_limit": PRIMARY_LIMIT,
            "representative_regression_limit": REGRESSION_LIMIT,
            "python_control_bounds": CONTROL_BOUNDS,
            "gate_policy_supplied": True,
            "focused_workloads": FOCUSED_WORKLOAD_NAMES,
            "full_representative_suite": True,
        },
        "baseline": baseline_environment,
        "candidate": candidate_environment,
        "trace_digest": attribution._checksum(manifest["workloads"][0]),
        "representative_workload_manifest": manifest,
        "semantic_evidence": semantic_evidence,
        "round_evidence": rounds,
        "semantic_checksums_match": True,
        "environments_match": True,
        "binary_provenance_complete": True,
        "clean_worktrees": not dirty,
        "controls": {
            "round_geomean_ratios": [1.0] * 20,
            "median_geomean_ratio": 1.0,
            "valid": True,
        },
        "layers": {
            name: primary if name == PRIMARY_LAYER else regression
            for name in REQUIRED_LAYERS
        },
        "focused": {name: regression for name in FOCUSED_WORKLOAD_NAMES},
        "representative": {name: regression for name in representative_names},
        "status": status,
        "promotion_eligible": not dirty,
    }


def _verify_gate(report: dict[str, Any]) -> None:
    verify_report(
        report,
        gate=True,
        expected_primary_ratio_limit=PRIMARY_LIMIT,
        expected_regression_ratio_limit=REGRESSION_LIMIT,
        expected_control_ratio_bounds=CONTROL_BOUNDS,
    )


def test_paired_ratio_gate_distinguishes_pass_fail_and_inconclusive() -> None:
    passed = paired_statistics(_samples(100), _samples(80), ratio_limit=0.85)
    failed = paired_statistics(_samples(100), _samples(90), ratio_limit=0.85)
    inconclusive = paired_statistics(
        _samples(100), [80] * 11 + [90] * 9, ratio_limit=0.85
    )
    diagnostic = paired_statistics(_samples(100), _samples(80), ratio_limit=None)

    assert passed["classification"] == "pass"
    assert passed["median_improvement"] == pytest.approx(0.20)
    assert failed["classification"] == "fail"
    assert inconclusive["classification"] == "inconclusive"
    assert diagnostic["classification"] == "not-evaluated"


@pytest.mark.parametrize(
    ("baseline", "candidate", "message"),
    [
        ([100] * 19, [80] * 19, "at least 20"),
        (_samples(100), [80] * 19, "equal length"),
        (_samples(100), [0] * 20, "positive"),
    ],
)
def test_paired_ratio_gate_rejects_invalid_samples(
    baseline: list[int], candidate: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        paired_statistics(baseline, candidate, ratio_limit=0.85)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("paired_ratios", [0.1] * 20, "inconsistent paired_ratios"),
        ("median_improvement", 0.9, "inconsistent median_improvement"),
        ("median_ratio", 0.1, "inconsistent median_ratio"),
    ],
)
def test_verifier_recomputes_all_derived_comparison_fields(
    field: str, value: object, message: str
) -> None:
    report = _passing_report()
    report["layers"][PRIMARY_LAYER][field] = value

    with pytest.raises(ValueError, match=message):
        verify_report(report)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "positive signed 64-bit integer samples"),
        ([80] * 19, "samples differ from per-round evidence"),
        ([80.0] * 20, "positive signed 64-bit integer samples"),
    ],
)
def test_verifier_rejects_missing_mistyped_or_miscounted_sample_arrays(
    value: object, message: str
) -> None:
    report = _passing_report()
    report["layers"][PRIMARY_LAYER]["candidate_samples_ns"] = value

    with pytest.raises(ValueError, match=message):
        verify_report(report)


@pytest.mark.parametrize("replacement", [None, {}, {"schema": "arbitrary"}])
def test_verifier_rejects_missing_or_arbitrary_representative_manifest(
    replacement: object,
) -> None:
    report = _passing_report()
    report["representative_workload_manifest"] = replacement

    with pytest.raises(ValueError, match="manifest"):
        verify_report(report)


def test_verifier_rejects_missing_workload_and_per_round_evidence() -> None:
    report = _passing_report()
    report["representative"].pop(next(iter(report["representative"])))
    with pytest.raises(ValueError, match="representative comparisons"):
        verify_report(report)

    report = _passing_report()
    report["round_evidence"].pop()
    with pytest.raises(ValueError, match="per-round evidence"):
        verify_report(report)


@pytest.mark.parametrize(
    "replacement",
    [None, {}, {"state_checksum": "c" * 64}],
)
def test_verifier_rejects_null_or_incomplete_per_round_semantics(
    replacement: object,
) -> None:
    report = copy.deepcopy(_passing_report())
    report["round_evidence"][1]["baseline"]["semantic"]["layers"][PRIMARY_LAYER] = (
        replacement
    )

    with pytest.raises(ValueError, match="semantics"):
        verify_report(report)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("state_checksum", None),
        ("operation_count", True),
        ("unexpected", 1),
    ],
)
def test_verifier_rejects_invalid_or_unexpected_semantic_values(
    field: str, replacement: object
) -> None:
    report = copy.deepcopy(_passing_report())
    entry = report["round_evidence"][1]["baseline"]["semantic"]["layers"][PRIMARY_LAYER]
    entry[field] = replacement

    with pytest.raises(ValueError, match="semantics"):
        verify_report(report)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("commit", "c" * 40),
        ("extension_path", "treemendous/cpp/replaced-boundary.so"),
        ("extension_sha256", "a" * 64),
        ("compiler_invocations", "c++ -c changed-boundary.cpp"),
    ],
)
def test_verifier_rejects_per_root_provenance_changes_between_rounds(
    field: str, replacement: str
) -> None:
    report = copy.deepcopy(_passing_report())
    report["round_evidence"][1]["baseline"]["environment"][field] = replacement

    with pytest.raises(ValueError, match=rf"{field} changed between rounds"):
        verify_report(report)


@pytest.mark.parametrize("commit", ["unknown", "a" * 39, "g" * 40])
def test_malformed_commit_is_diagnostic_and_never_promotion_eligible(
    commit: str,
) -> None:
    report = _passing_report()
    report["baseline"]["commit"] = commit
    for round_entry in report["round_evidence"]:
        round_entry["baseline"]["environment"]["commit"] = commit
    report["status"] = "diagnostic"
    report["promotion_eligible"] = False

    verify_report(report)
    with pytest.raises(ValueError, match="not promotion-eligible"):
        _verify_gate(report)


def test_build_provenance_normalizes_shared_python_environment_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_root = tmp_path / "baseline"
    candidate_root = Path(attribution.sys.prefix).resolve().parent

    def fake_run(
        command: list[str], *, cwd: Path, check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        assert check and capture_output and text
        compiler_output = (
            f"clang++ -I{attribution.sys.prefix}/include "
            f"-c {cwd}/treemendous/cpp/boundary_bindings.cpp "
            "-o build/boundary.o\n"
        )
        return subprocess.CompletedProcess(command, 0, compiler_output, "")

    monkeypatch.setattr(attribution.subprocess, "run", fake_run)

    baseline = attribution._build_root(baseline_root)
    candidate = attribution._build_root(candidate_root)

    assert baseline == candidate
    assert "<PYTHON_ENV>/include" in baseline["compiler_invocations"]
    assert (
        str(Path(attribution.sys.prefix).resolve())
        not in baseline["compiler_invocations"]
    )


def test_failed_git_status_is_recorded_as_unclean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commit = "a" * 40

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(
                command, 0, stdout=f"{commit}\n", stderr=""
            )
        if command[:3] == ["git", "status", "--porcelain"]:
            raise subprocess.CalledProcessError(1, command)
        raise AssertionError(f"unexpected subprocess command: {command}")

    monkeypatch.setattr(attribution, "_compiler_version", lambda: "c++ 1.0")
    monkeypatch.setattr(attribution.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(attribution.platform, "machine", lambda: "test-machine")
    monkeypatch.setattr(attribution.platform, "processor", lambda: "test-processor")
    monkeypatch.setattr(
        attribution,
        "_native_extension_provenance",
        lambda source_root: {
            "extension_path": "treemendous/cpp/boundary.so",
            "extension_sha256": "e" * 64,
        },
    )
    monkeypatch.setattr(attribution.subprocess, "run", fake_run)

    environment = attribution._environment(tmp_path)

    assert environment["commit"] == commit
    assert environment["dirty"] == "true"


def test_dirty_diagnostic_is_valid_but_cannot_be_gated_or_bypassed() -> None:
    dirty = _passing_report(dirty=True)
    verify_report(dirty)
    with pytest.raises(ValueError, match="not promotion-eligible"):
        _verify_gate(dirty)
    with pytest.raises(TypeError, match="allow_dirty"):
        verify_report(dirty, allow_dirty=True)  # type: ignore[call-arg]


def test_policy_failure_is_verifiable_diagnostic_but_fails_explicit_gate() -> None:
    report = _passing_report()
    failed = paired_statistics(
        _samples(100), _samples(110), ratio_limit=REGRESSION_LIMIT
    )
    report["layers"]["observed_publication"] = failed
    for round_entry in report["round_evidence"]:
        round_entry["candidate"]["timings_ns"]["layers"]["observed_publication"] = 110
    report["status"] = "fail"
    report["promotion_eligible"] = False

    verify_report(report)
    with pytest.raises(ValueError, match="not promotion-eligible"):
        _verify_gate(report)


def test_attribution_artifact_triplet_and_duplicate_keys(tmp_path: Path) -> None:
    output = tmp_path / "attribution.json"
    write_artifacts(_passing_report(), output)
    verified = verify_artifact(
        output,
        expected_baseline="a" * 40,
        expected_candidate="b" * 40,
    )
    assert verified["status"] == "pass"

    encoded = (
        output.read_text(encoding="utf-8")
        .replace(
            f'"schema": "{SCHEMA}",',
            f'"schema": "{SCHEMA}",\n  "schema": "duplicate",',
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


def test_compare_roots_status_uses_explicit_policy_and_dirty_is_diagnostic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    manifest = representative_workload_manifest()
    names = tuple(item["name"] for item in manifest["workloads"])
    commits = {"baseline": "a" * 40, "candidate": "b" * 40}

    monkeypatch.setattr(
        attribution,
        "_build_root",
        lambda source_root: {
            "build_command": "${PYTHON} setup.py build_ext --inplace --force --verbose",
            "compiler_invocations": "c++ -c boundary.cpp",
        },
    )

    def fake_worker(source_root: Path, *, warmups: int, full: bool) -> dict[str, Any]:
        del warmups
        candidate = source_root == candidate_root.resolve()

        def timed(value: int) -> dict[str, Any]:
            return {
                "execution_ns": value,
                "state_checksum": "s",
                "query_checksum": "q",
                "operation_count": 1_000,
                "successful_operation_count": 1_000,
                "no_op_count": 0,
                "touched_interval_count": 1_000,
                "touched_length": 1_000,
            }

        label = "candidate" if candidate else "baseline"
        environment = _environment(commits[label])
        return {
            "environment": environment,
            "trace_digest": attribution._checksum(manifest["workloads"][0]),
            "representative_manifest": manifest,
            "layers": {
                name: timed(80 if candidate and name == PRIMARY_LAYER else 100)
                for name in REQUIRED_LAYERS
            },
            "focused": {name: timed(100) for name in FOCUSED_WORKLOAD_NAMES if full},
            "representative": {name: timed(100) for name in names if full},
            "controls": {name: timed(100) for name in PYTHON_CONTROLS if full},
        }

    monkeypatch.setattr(attribution, "_run_worker_process", fake_worker)
    diagnostic = attribution.compare_roots(
        baseline_root, candidate_root, samples=20, warmups=1, full=True
    )
    assert diagnostic["status"] == "diagnostic"
    assert not diagnostic["promotion_eligible"]

    passed = attribution.compare_roots(
        baseline_root,
        candidate_root,
        samples=20,
        warmups=1,
        full=True,
        primary_ratio_limit=PRIMARY_LIMIT,
        regression_ratio_limit=REGRESSION_LIMIT,
        control_ratio_bounds=CONTROL_BOUNDS,
    )
    assert passed["status"] == "pass"
    assert passed["promotion_eligible"]

    commits["baseline"] = "unknown"
    malformed_commit = attribution.compare_roots(
        baseline_root,
        candidate_root,
        samples=20,
        warmups=1,
        full=True,
        primary_ratio_limit=PRIMARY_LIMIT,
        regression_ratio_limit=REGRESSION_LIMIT,
        control_ratio_bounds=CONTROL_BOUNDS,
    )
    assert malformed_commit["status"] == "diagnostic"
    assert not malformed_commit["promotion_eligible"]
