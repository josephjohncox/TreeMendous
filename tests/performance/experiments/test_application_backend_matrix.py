"""Focused contracts for the concrete application backend experiment."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.performance.experiments import application_backend_matrix as experiment
from treemendous.backends.registry import BackendRegistry


def _rewrite(output: Path, report: dict[str, Any] | str) -> None:
    text = (
        report
        if isinstance(report, str)
        else json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    encoded = text.encode()
    output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    Path(f"{output}.sha256").write_text(f"{digest}  {output.name}\n")


def test_all_qualified_backends_match_exact_application_evidence_and_factories() -> (
    None
):
    registry = BackendRegistry.discover()
    specs = experiment._eligible_specs(registry)
    rows, _ = experiment._correctness_rows(registry, specs, experiment.ENGINE_NAMES)

    assert {row["backend"] for row in rows} == {spec.id for spec in specs}
    expected_calls = {
        "contiguous_allocator": 2,
        "disk_block_allocator": 2,
        "pool_group": 8,
        "claim_ledger": 14,
        "partition_runtime": 22,
    }
    baseline = {
        row["engine"]: row["semantic"]
        for row in rows
        if row["backend"] == experiment.BASELINE_BACKEND
    }
    for row in rows:
        assert row["semantic"] == baseline[row["engine"]]
        assert row["semantic_sha256"] == row["baseline_sha256"]
        assert row["factory"]["calls"] == expected_calls[row["engine"]]
        assert (
            len(set(row["factory"]["implementation_ids"]))
            == expected_calls[row["engine"]]
        )


def test_negative_requests_fail_closed_without_fallback() -> None:
    rows = experiment._negative_evidence(BackendRegistry.discover())

    assert [row["case"] for row in rows] == [
        "unknown",
        "unavailable",
        "invalid",
        "experimental",
        "32-bit",
        "nondeterministic",
    ]
    assert all(row["accepted"] is False for row in rows)
    assert all(row["fallback_backend"] is None for row in rows)
    for case in ("experimental", "32-bit", "nondeterministic"):
        row = next(item for item in rows if item["case"] == case)
        expected = {
            "experimental": "experimental",
            "32-bit": "signed 64-bit",
            "nondeterministic": "nondeterministic",
        }[case]
        assert expected in row["reason"]


@pytest.fixture(scope="module")
def focused_report() -> dict[str, Any]:
    return experiment.run_matrix(
        blocks=experiment.MINIMUM_BLOCKS,
        backend_ids=(experiment.BASELINE_BACKEND,),
    )


def test_bounded_balanced_matrix_and_strict_triplet(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output = tmp_path / "application-backends.json"
    paths = experiment.write_artifacts(focused_report, output)
    verified = experiment.verify_artifacts(output)

    assert verified == focused_report
    assert verified["gate"]["runtime_injection_retained"] is False
    assert len(verified["benchmarks"]) == len(experiment.ENGINE_NAMES)
    for row in verified["benchmarks"]:
        assert row["validated_blocks"] == experiment.MINIMUM_BLOCKS
        assert len(row["blocks"]) == experiment.MINIMUM_BLOCKS
        assert [block["order"] for block in row["blocks"][:2]] == [
            ["candidate", "baseline"],
            ["baseline", "candidate"],
        ]
    assert all(path.is_file() for path in paths)


def test_verifier_rejects_duplicate_nonfinite_and_exact_type_tamper(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output = tmp_path / "application-backends.json"
    experiment.write_artifacts(focused_report, output)

    duplicate = output.read_text().replace(
        f'  "schema": "{experiment.SCHEMA}"',
        f'  "schema": "duplicate",\n  "schema": "{experiment.SCHEMA}"',
    )
    _rewrite(output, duplicate)
    with pytest.raises(ValueError, match="duplicate key"):
        experiment.verify_artifacts(output)

    nonfinite = copy.deepcopy(focused_report)
    nonfinite["benchmarks"][0]["blocks"][0]["ratio"] = float("nan")
    _rewrite(output, json.dumps(nonfinite, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="non-finite"):
        experiment.verify_artifacts(output)

    wrong_type = copy.deepcopy(focused_report)
    wrong_type["benchmarks"][0]["blocks"][0]["candidate_ns"] = True
    experiment.write_artifacts(wrong_type, output)
    with pytest.raises(ValueError, match="duration exact type"):
        experiment.verify_artifacts(output)


def test_verifier_rejects_duplicate_correctness_benchmark_and_confirmation_rows(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output = tmp_path / "application-backends.json"

    correctness = copy.deepcopy(focused_report)
    correctness["correctness"].insert(1, copy.deepcopy(correctness["correctness"][0]))
    correctness["correctness"].pop()
    experiment.write_artifacts(correctness, output)
    with pytest.raises(ValueError, match="correctness matrix order/uniqueness"):
        experiment.verify_artifacts(output)

    benchmark = copy.deepcopy(focused_report)
    benchmark["benchmarks"].insert(1, copy.deepcopy(benchmark["benchmarks"][0]))
    benchmark["benchmarks"].pop()
    experiment.write_artifacts(benchmark, output)
    with pytest.raises(ValueError, match="benchmark matrix order/uniqueness"):
        experiment.verify_artifacts(output)

    confirmation = copy.deepcopy(focused_report)
    duplicate = copy.deepcopy(confirmation["benchmarks"][0])
    duplicate["phase"] = "confirmation"
    confirmation["confirmation"] = [duplicate, copy.deepcopy(duplicate)]
    experiment.write_artifacts(confirmation, output)
    with pytest.raises(ValueError, match="confirmation matrix order/uniqueness"):
        experiment.verify_artifacts(output)


def test_verifier_recomputes_ratios_gate_factory_and_provenance(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output = tmp_path / "application-backends.json"

    wrong_ratio = copy.deepcopy(focused_report)
    wrong_ratio["benchmarks"][0]["ratio"]["median_95_high"] += 0.01
    experiment.write_artifacts(wrong_ratio, output)
    with pytest.raises(ValueError, match="derived ratio interval"):
        experiment.verify_artifacts(output)

    wrong_gate = copy.deepcopy(focused_report)
    wrong_gate["gate"]["runtime_injection_retained"] = True
    experiment.write_artifacts(wrong_gate, output)
    with pytest.raises(ValueError, match="recomputed gate"):
        experiment.verify_artifacts(output)

    wrong_factory = copy.deepcopy(focused_report)
    wrong_factory["correctness"][0]["factory"]["calls"] += 1
    experiment.write_artifacts(wrong_factory, output)
    with pytest.raises(ValueError, match="factory call/identity"):
        experiment.verify_artifacts(output)

    wrong_source = copy.deepcopy(focused_report)
    wrong_source["provenance"]["sources"][0]["sha256"] = "0" * 64
    experiment.write_artifacts(wrong_source, output)
    with pytest.raises(ValueError, match="provenance"):
        experiment.verify_artifacts(output)

    experiment.write_artifacts(focused_report, output)
    output.with_suffix(".md").write_text("unbound\n")
    with pytest.raises(ValueError, match="Markdown mismatch"):
        experiment.verify_artifacts(output)
