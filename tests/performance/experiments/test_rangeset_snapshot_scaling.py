"""Focused strict checks for the geometry snapshot scaling experiment."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.performance.experiments import rangeset_snapshot_scaling as experiment
from treemendous import RangeSnapshot, Span, create_range_set


def test_faithful_uncached_construction_matches_without_public_identity_reuse() -> None:
    ranges = create_range_set((0, 30), backend="py_boundary", initially_available=False)
    ranges.add(Span(2, 5))
    ranges.add(Span(10, 14))
    cached = ranges.snapshot()
    first = experiment._uncached_snapshot(ranges)
    second = experiment._uncached_snapshot(ranges)

    assert type(cached) is RangeSnapshot
    assert type(first) is RangeSnapshot
    assert cached == first == second
    assert cached is ranges.snapshot()
    assert first is not cached and second is not cached and first is not second


def _synthetic_row(kind: str, count: int, blocks: int) -> dict[str, Any]:
    iterations = experiment.PILOT_MIN_ITERATIONS
    raw_blocks: list[dict[str, Any]] = []
    cached_totals: list[int] = []
    uncached_totals: list[int] = []
    for block_index in range(blocks):
        cached_base = 20_000 if kind == "write_then_observe_restorative" else 1_000
        cached_values = [cached_base + block_index, cached_base + block_index + 10]
        uncached_values = [10_000 + block_index, 10_020 + block_index]
        orderings = (
            ["cached_first", "uncached_first"]
            if block_index % 2 == 0
            else ["uncached_first", "cached_first"]
        )
        positions = [
            {
                "ordering": ordering,
                "cached_ns": cached_value,
                "uncached_ns": uncached_value,
            }
            for ordering, cached_value, uncached_value in zip(
                orderings, cached_values, uncached_values, strict=True
            )
        ]
        cached_total = sum(cached_values)
        uncached_total = sum(uncached_values)
        raw_blocks.append(
            {
                "block_index": block_index,
                "positions": positions,
                "cached_ns_total": cached_total,
                "uncached_ns_total": uncached_total,
                "ratio": cached_total / uncached_total,
            }
        )
        cached_totals.append(cached_total)
        uncached_totals.append(uncached_total)
    seed = experiment.BOOTSTRAP_SEED + count * (
        1 if kind == "unchanged_read_burst" else 2
    )
    ratios, ratio = experiment._ratio_summary(cached_totals, uncached_totals, seed)
    divisor = 2 * iterations
    return {
        "kind": kind,
        "interval_count": count,
        "operations_per_iteration": (
            experiment.UNCHANGED_READS if kind == "unchanged_read_burst" else 4
        ),
        "iterations_per_position": iterations,
        "pilot": {
            "target_position_ns": experiment.PILOT_TARGET_NS,
            "maximum_iterations": experiment.PILOT_MAX_ITERATIONS,
            "order": "cached_then_uncached",
            "measurements": [
                {
                    "iterations": iterations,
                    "cached_ns": experiment.PILOT_TARGET_NS + 1,
                    "uncached_ns": experiment.PILOT_TARGET_NS + 2,
                }
            ],
            "selected_iterations": iterations,
            "target_reached_by_both": True,
            "excluded_from_blocks_and_gates": True,
        },
        "blocks": raw_blocks,
        "block_ratios": ratios,
        "cached_ns_per_iteration": experiment._summary(
            [value / divisor for value in cached_totals], seed + 1
        ),
        "uncached_ns_per_iteration": experiment._summary(
            [value / divisor for value in uncached_totals], seed + 2
        ),
        "ratio": ratio,
        "validated_blocks": blocks,
        "final_snapshot_sha256": experiment._expected_snapshot_digest(count),
    }


@pytest.fixture(scope="module")
def focused_report() -> dict[str, Any]:
    blocks = experiment.CONFIRMATION_BLOCKS
    rows = [
        _synthetic_row(kind, count, blocks)
        for count in experiment.INTERVAL_COUNTS
        for kind in ("unchanged_read_burst", "write_then_observe_restorative")
    ]
    return {
        "schema": experiment.SCHEMA,
        "blocks": blocks,
        "methodology": experiment._methodology(),
        "rows": rows,
        "gate": experiment._gate(rows),
        "provenance": experiment._provenance(),
    }


def test_focused_matrix_has_balanced_raw_blocks_and_exact_derivations(
    focused_report: dict[str, Any],
) -> None:
    assert focused_report["gate"]["decision"] == "REJECTED"
    assert focused_report["blocks"] == 40
    assert focused_report["blocks"] >= experiment.MINIMUM_BLOCKS
    for row in focused_report["rows"]:
        assert row["validated_blocks"] == experiment.CONFIRMATION_BLOCKS
        assert row["iterations_per_position"] >= 2
        assert row["pilot"]["excluded_from_blocks_and_gates"] is True
        assert len(row["blocks"]) == experiment.CONFIRMATION_BLOCKS
        assert len(row["block_ratios"]) == experiment.CONFIRMATION_BLOCKS
        for block in row["blocks"]:
            assert {position["ordering"] for position in block["positions"]} == {
                "cached_first",
                "uncached_first",
            }
            assert block["cached_ns_total"] == sum(
                position["cached_ns"] for position in block["positions"]
            )
            assert block["uncached_ns_total"] == sum(
                position["uncached_ns"] for position in block["positions"]
            )
            assert block["ratio"] == (
                block["cached_ns_total"] / block["uncached_ns_total"]
            )


def test_artifact_triplet_round_trips_through_strict_verifier(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output = tmp_path / "snapshot-scaling.json"
    paths = experiment.write_artifacts(focused_report, output)
    verified = experiment.verify_artifacts(output)
    assert verified == focused_report
    assert all(path.is_file() for path in paths)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    assert digest in output.with_suffix(".md").read_text()


def _rewrite(output: Path, text: str) -> None:
    encoded = text.encode()
    output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    Path(f"{output}.sha256").write_text(f"{digest}  {output.name}\n")


def test_verifier_rejects_duplicate_nonfinite_and_exact_type_tampering(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output = tmp_path / "snapshot-scaling.json"
    experiment.write_artifacts(focused_report, output)
    duplicate = output.read_text().replace(
        f'  "schema": "{experiment.SCHEMA}"',
        f'  "schema": "duplicate",\n  "schema": "{experiment.SCHEMA}"',
        1,
    )
    _rewrite(output, duplicate)
    with pytest.raises(ValueError, match="duplicate key"):
        experiment.verify_artifacts(output)

    nonfinite = copy.deepcopy(focused_report)
    nonfinite["rows"][0]["blocks"][0]["ratio"] = float("nan")
    _rewrite(output, json.dumps(nonfinite, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="non-finite"):
        experiment.verify_artifacts(output)

    wrong_type = copy.deepcopy(focused_report)
    wrong_type["rows"][0]["blocks"][0]["positions"][0]["cached_ns"] = True
    experiment.write_artifacts(wrong_type, output)
    with pytest.raises(ValueError, match="balanced block ordering/duration"):
        experiment.verify_artifacts(output)


def test_verifier_rejects_block_order_iterations_totals_and_derivations(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output = tmp_path / "snapshot-scaling.json"

    wrong_order = copy.deepcopy(focused_report)
    wrong_order["rows"][0]["blocks"][0]["positions"][0]["ordering"] = "uncached_first"
    experiment.write_artifacts(wrong_order, output)
    with pytest.raises(ValueError, match="balanced block ordering"):
        experiment.verify_artifacts(output)

    wrong_iterations = copy.deepcopy(focused_report)
    wrong_iterations["rows"][0]["iterations_per_position"] = 3
    experiment.write_artifacts(wrong_iterations, output)
    with pytest.raises(ValueError, match="pilot fixed methodology"):
        experiment.verify_artifacts(output)

    wrong_pilot_duration = copy.deepcopy(focused_report)
    wrong_pilot_duration["rows"][0]["pilot"]["measurements"][0]["cached_ns"] = 0
    experiment.write_artifacts(wrong_pilot_duration, output)
    with pytest.raises(ValueError, match="pilot duration/iteration"):
        experiment.verify_artifacts(output)

    wrong_total = copy.deepcopy(focused_report)
    wrong_total["rows"][0]["blocks"][0]["cached_ns_total"] += 1
    experiment.write_artifacts(wrong_total, output)
    with pytest.raises(ValueError, match="raw block total/ratio"):
        experiment.verify_artifacts(output)

    wrong_ratio = copy.deepcopy(focused_report)
    wrong_ratio["rows"][0]["ratio"]["median_95_high"] += 0.01
    experiment.write_artifacts(wrong_ratio, output)
    with pytest.raises(ValueError, match="derived block ratio interval"):
        experiment.verify_artifacts(output)


def test_verifier_recomputes_gates_snapshot_and_markdown(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output = tmp_path / "snapshot-scaling.json"

    wrong_gate = copy.deepcopy(focused_report)
    wrong_gate["gate"]["decision"] = "ACCEPTED"
    experiment.write_artifacts(wrong_gate, output)
    with pytest.raises(ValueError, match="gate mismatch"):
        experiment.verify_artifacts(output)

    wrong_snapshot = copy.deepcopy(focused_report)
    wrong_snapshot["rows"][0]["final_snapshot_sha256"] = "0" * 64
    experiment.write_artifacts(wrong_snapshot, output)
    with pytest.raises(ValueError, match="snapshot digest semantic mismatch"):
        experiment.verify_artifacts(output)

    experiment.write_artifacts(focused_report, output)
    output.with_suffix(".md").write_text("unbound report\n")
    with pytest.raises(ValueError, match="Markdown mismatch"):
        experiment.verify_artifacts(output)


@pytest.mark.parametrize(
    "path",
    (
        ("runtime", "python_version"),
        ("runtime", "python_implementation"),
        ("runtime", "python_compiler"),
        ("runtime", "platform"),
        ("runtime", "machine"),
        ("runtime", "architecture"),
        ("build", "command"),
        ("build", "cxx"),
        ("build", "cxx_version"),
        ("build", "cc"),
        ("build", "cflags"),
        ("build", "flags", "TREE_MENDOUS_WITH_ICL"),
        ("backend", "id"),
        ("backend", "module"),
        ("backend", "type"),
        ("backend", "path"),
        ("backend", "sha256"),
    ),
)
def test_verifier_rejects_each_runtime_build_and_backend_provenance_category(
    tmp_path: Path, focused_report: dict[str, Any], path: tuple[str, ...]
) -> None:
    output = tmp_path / "snapshot-scaling.json"
    tampered = copy.deepcopy(focused_report)
    target = tampered["provenance"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = True
    experiment.write_artifacts(tampered, output)

    with pytest.raises(ValueError, match="runtime|build|active backend"):
        experiment.verify_artifacts(output)


def test_verifier_rejects_exact_methodology_and_provenance_hash_tamper(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output = tmp_path / "snapshot-scaling.json"

    wrong_method = copy.deepcopy(focused_report)
    wrong_method["methodology"]["balanced_block_method"] = "tampered"
    experiment.write_artifacts(wrong_method, output)
    with pytest.raises(ValueError, match="fixed matrix/methodology"):
        experiment.verify_artifacts(output)

    wrong_block_count = copy.deepcopy(focused_report)
    wrong_block_count["blocks"] = experiment.MINIMUM_BLOCKS
    experiment.write_artifacts(wrong_block_count, output)
    with pytest.raises(ValueError, match="fixed block count"):
        experiment.verify_artifacts(output)

    wrong_source_state = copy.deepcopy(focused_report)
    wrong_source_state["provenance"]["git"]["source_state_sha256"] = "0" * 64
    experiment.write_artifacts(wrong_source_state, output)
    with pytest.raises(ValueError, match="git/source-state provenance"):
        experiment.verify_artifacts(output)

    wrong_source_file = copy.deepcopy(focused_report)
    wrong_source_file["provenance"]["sources"][0]["sha256"] = "0" * 64
    experiment.write_artifacts(wrong_source_file, output)
    with pytest.raises(ValueError, match="source provenance path/hash"):
        experiment.verify_artifacts(output)
