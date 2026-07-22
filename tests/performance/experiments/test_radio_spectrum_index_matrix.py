"""Focused contracts for the radio-spectrum index representation experiment."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.performance.experiments import radio_spectrum_index_matrix as experiment
from treemendous.applications.scheduling.radio_spectrum import RadioSpectrumScheduler
from treemendous.multidimensional import BoxIndex


def _rewrite(output: Path, report: dict[str, Any] | str) -> None:
    text = (
        report
        if isinstance(report, str)
        else json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    encoded = text.encode()
    output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    Path(f"{output}.sha256").write_text(f"{digest}  {output.name}\n")


def test_private_empty_scheduler_injection_does_not_change_runtime_default() -> None:
    stable = RadioSpectrumScheduler(experiment.CHANNEL_COUNT)
    projected = experiment._new_scheduler(
        "projection",
        active_entries=32,
        blocks=experiment.MINIMUM_BLOCKS,
        scenario_count=1,
    )

    assert type(stable._index) is BoxIndex
    assert stable._index.diagnostics().algorithm == "linear"
    assert projected._index.diagnostics().algorithm == "axis_projection"
    assert not stable._records and not projected._records


def test_independent_oracle_covers_identity_conflicts_cancellation_and_state() -> None:
    traces = [
        experiment._semantic_trace(kind) for kind in ("linear", "projection", "grid")
    ]

    normalized = []
    for trace in traces:
        state = copy.deepcopy(trace["final_state"])
        state["diagnostics"].update(
            {
                "algorithm": "normalized",
                "projection_sizes": [],
                "posting_count": None,
                "occupied_cell_count": None,
                "estimated_memory_bytes": None,
            }
        )
        normalized.append(
            {
                "calls": trace["calls"],
                "snapshot_order": trace["snapshot_order"],
                "snapshot_version": trace["snapshot_version"],
                "final_state": state,
            }
        )
    assert normalized[0] == normalized[1] == normalized[2]
    assert traces[0]["calls"][1]["same_identity"] is True
    assert traces[0]["calls"][7]["same_identity"] is True
    assert traces[0]["snapshot_version"] == 4


def test_grid_guard_error_propagates_without_fallback_or_mutation() -> None:
    evidence = experiment._grid_guard_adversary()

    assert evidence["algorithm"] == "sparse_grid"
    assert evidence["fallback"] is None
    assert evidence["state_unchanged"] is True
    assert "max_cells_per_query" in evidence["error"]["message"]
    assert evidence["limits"]["max_cells_per_query"] == 64


def test_density_and_axis_skew_change_seed_and_oracle_query_geometry() -> None:
    base = experiment.SCENARIOS[0]
    high_density = dataclasses.replace(base, density="high")
    channel_skew = dataclasses.replace(base, axis_skew="channel")
    assert experiment._seed_arguments(base, 40) != experiment._seed_arguments(
        high_density, 40
    )
    assert experiment._seed_arguments(base, 40) != experiment._seed_arguments(
        channel_skew, 40
    )
    low_oracle = experiment._seed_oracle(base, 128)
    high_oracle = experiment._seed_oracle(high_density, 128)
    low_query = low_oracle.conflicts_for(*experiment._query_arguments(base, 128, 40))
    high_query = high_oracle.conflicts_for(
        *experiment._query_arguments(high_density, 128, 40)
    )
    assert low_query is not None and high_query is not None
    assert len(low_query[1]) == 1
    assert len(high_query[1]) > len(low_query[1])

    scenarios = experiment.SCENARIOS[:4]
    seed_shapes = {
        scenario.name: tuple(
            tuple(
                experiment._seed_arguments(scenario, index)[key]
                for key in ("channel_start", "start", "end")
            )
            for index in (0, 40, 100)
        )
        for scenario in scenarios
    }
    assert len(set(seed_shapes.values())) == len(scenarios)

    match_counts = {}
    for scenario in scenarios:
        oracle = experiment._seed_oracle(scenario, 128)
        counts = []
        for query_index in range(0, 112, 7):
            arguments = experiment._query_arguments(scenario, 128, query_index)
            conflict = oracle.conflicts_for(*arguments)
            counts.append(0 if conflict is None else len(conflict[1]))
        match_counts[scenario.name] = counts
    assert set(match_counts["low-narrow-narrow"]) == {1}
    assert min(match_counts["medium-broad-channel"]) > 1
    assert min(match_counts["medium-broad-time"]) > 1
    assert set(match_counts["high-broad-both"]) == {128}


def test_mutation_mix_uses_distinct_reservations_and_separate_replay() -> None:
    by_ratio = {
        scenario.insertion_cancellation_ratio: scenario
        for scenario in experiment.SCENARIOS
    }
    for ratio in ("3:1", "2:2", "1:3"):
        setup, timed = experiment._commands(
            by_ratio[ratio],
            active_entries=128,
            scenario_index=0,
            block_index=0,
            total_blocks=experiment.MINIMUM_BLOCKS,
            seed=1,
        )
        reserve_calls = [
            command for command in (*setup, *timed) if command[0] == "reserve"
        ]
        request_ids = [command[2]["request_id"] for command in reserve_calls]
        assert len(request_ids) == len(set(request_ids))
        cancellations = [command[1][1] for command in timed if command[0] == "cancel"]
        assert len(cancellations) == len(set(cancellations))

    replay = by_ratio["replay"]
    assert replay.name == "idempotent-replay"
    setup, timed = experiment._commands(
        replay,
        active_entries=128,
        scenario_index=0,
        block_index=0,
        total_blocks=experiment.MINIMUM_BLOCKS,
        seed=1,
    )
    assert len(setup) == 1
    assert len({(command[1], tuple(command[2].items())) for command in timed[4:]}) == 1


def _gate_evidence(
    held_out_upper: float | None,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    scenario = experiment.SCENARIOS[0]

    def row(phase: str, size: int, upper: float) -> dict[str, Any]:
        return {
            "phase": phase,
            "candidate": "projection",
            "scenario": {"name": scenario.name},
            "active_entries": size,
            "ratio": {"median_95_high": upper},
        }

    training = [row("training", 2_000, 0.75), row("training", 10_000, 0.75)]
    held_out = (
        [] if held_out_upper is None else [row("held_out", 5_000, held_out_upper)]
    )
    construction = [
        {
            "phase": item["phase"],
            "candidate": item["candidate"],
            "scenario": item["scenario"],
            "active_entries": item["active_entries"],
            "retained_memory_ratio": 1.0,
        }
        for item in [*training, *held_out]
    ]
    control = {"ratio": {"median_95_high": 1.0}}
    return training, held_out, construction, control


def test_gate_uses_lower_adjacent_size_and_requires_held_out_confirmation() -> None:
    training, held_out, construction, control = _gate_evidence(0.95)
    decision = experiment._gate(
        training, held_out, construction, (2_000, 10_000), control
    )
    assert decision["decision"] == "REJECTED"
    assert any(
        "held-out selected cells failed" in reason
        for reason in decision["rejected_reasons"]
    )

    training, held_out, construction, control = _gate_evidence(None)
    decision = experiment._gate(
        training, held_out, construction, (2_000, 10_000), control
    )
    assert decision["decision"] == "REJECTED"
    assert any(
        "lack predetermined held-out evidence" in reason
        for reason in decision["rejected_reasons"]
    )

    training, held_out, construction, control = _gate_evidence(0.85)
    decision = experiment._gate(
        training, held_out, construction, (2_000, 10_000), control
    )
    assert decision["decision"] == "QUALIFIED_PRIVATE_CONFIRMATION_REQUIRED"
    assert "projection:low-narrow-narrow:2000" in decision["selected_cells"]
    assert "projection:low-narrow-narrow:5000" in decision["selected_cells"]

    control["ratio"]["median_95_high"] = 1.11
    decision = experiment._gate(
        training, held_out, construction, (2_000, 10_000), control
    )
    assert decision["decision"] == "REJECTED"
    assert decision["default_linear_upper_ratio"] == 1.11
    assert any(
        "explicitly patched linear control" in reason
        for reason in decision["rejected_reasons"]
    )


@pytest.fixture(scope="module")
def focused_report() -> dict[str, Any]:
    return experiment.run_matrix(
        profile="focused",
        blocks=experiment.MINIMUM_BLOCKS,
        training_sizes=experiment.TRAINING_SIZES[:2],
        held_out_sizes=experiment.HELD_OUT_SIZES[:1],
        scenarios=experiment.SCENARIOS[:1],
        candidates=("projection",),
    )


def test_balanced_blocks_whole_block_bootstrap_and_rejected_runtime(
    focused_report: dict[str, Any],
) -> None:
    assert focused_report["methodology"]["blocks_recorded_before_run"] == 25
    assert focused_report["decision"]["decision"] == "REJECTED"
    assert focused_report["decision"]["runtime_index"] == "linear"
    assert focused_report["decision"]["runtime_seam_retained"] is False
    assert focused_report["decision"]["live_migration"] is False
    assert focused_report["decision"]["selected_cells"] == []
    control = focused_report["default_linear_control"]
    assert control["candidate"] == "default"
    assert control["phase"] == "default_linear_control"
    assert (
        control["ratio"]["median_95_high"]
        == focused_report["decision"]["default_linear_upper_ratio"]
    )
    assert control["ratio"]["median"] != 1.0
    assert len(control["blocks"]) == experiment.MINIMUM_BLOCKS
    for row in [*focused_report["training"], *focused_report["held_out"], control]:
        assert row["validated_blocks"] == experiment.MINIMUM_BLOCKS
        assert row["operations_per_position"] == experiment.OPERATIONS_PER_POSITION
        assert [block["order"] for block in row["blocks"][:2]] == [
            ["candidate", "baseline"],
            ["baseline", "candidate"],
        ]
        expected_algorithm = "linear" if row is control else "axis_projection"
        assert row["query_diagnostics"]["algorithm"] == expected_algorithm


def test_canonical_triplet_and_strict_verifier(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output = tmp_path / "radio-spectrum.json"
    paths = experiment.write_artifacts(focused_report, output)

    assert experiment.verify_artifacts(output) == focused_report
    assert all(path.is_file() for path in paths)
    assert (
        output.with_suffix(".md")
        .read_text()
        .startswith("# RadioSpectrumScheduler index representation experiment\n")
    )


def test_verifier_rejects_duplicate_nonfinite_exact_type_and_derived_tamper(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output = tmp_path / "radio-spectrum.json"

    duplicate = (
        json.dumps(focused_report, indent=2, sort_keys=True).replace(
            f'  "schema": "{experiment.SCHEMA}"',
            f'  "schema": "duplicate",\n  "schema": "{experiment.SCHEMA}"',
        )
        + "\n"
    )
    experiment.write_artifacts(focused_report, output)
    _rewrite(output, duplicate)
    with pytest.raises(ValueError, match="duplicate key"):
        experiment.verify_artifacts(output)

    nonfinite = copy.deepcopy(focused_report)
    nonfinite["training"][0]["blocks"][0]["ratio"] = float("nan")
    text = json.dumps(nonfinite, indent=2, sort_keys=True) + "\n"
    _rewrite(output, text)
    with pytest.raises(ValueError, match="non-finite"):
        experiment.verify_artifacts(output)

    wrong_type = copy.deepcopy(focused_report)
    wrong_type["training"][0]["blocks"][0]["candidate_ns"] = True
    experiment.write_artifacts(wrong_type, output)
    with pytest.raises(ValueError, match="exact type"):
        experiment.verify_artifacts(output)

    wrong_ratio = copy.deepcopy(focused_report)
    wrong_ratio["training"][0]["ratio"]["median_95_high"] += 0.01
    experiment.write_artifacts(wrong_ratio, output)
    with pytest.raises(ValueError, match="derived ratio"):
        experiment.verify_artifacts(output)

    wrong_decision = copy.deepcopy(focused_report)
    wrong_decision["decision"]["runtime_seam_retained"] = True
    experiment.write_artifacts(wrong_decision, output)
    with pytest.raises(ValueError, match="recomputed decision"):
        experiment.verify_artifacts(output)


def test_verifier_rejects_row_relabel_duplicate_diagnostics_and_digest_tamper(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output = tmp_path / "radio-spectrum.json"

    relabel = copy.deepcopy(focused_report)
    relabel["training"][0]["phase"] = "held_out"
    experiment.write_artifacts(focused_report, output)
    _rewrite(output, relabel)
    with pytest.raises(ValueError, match="benchmark row exact type/value"):
        experiment.verify_artifacts(output)

    duplicate = copy.deepcopy(focused_report)
    duplicate["training"][1] = copy.deepcopy(duplicate["training"][0])
    experiment.write_artifacts(focused_report, output)
    _rewrite(output, duplicate)
    with pytest.raises(ValueError, match="benchmark row exact type/value"):
        experiment.verify_artifacts(output)

    diagnostics = copy.deepcopy(focused_report)
    diagnostics["training"][0]["query_diagnostics"]["candidate_median"] += 1.0
    experiment.write_artifacts(focused_report, output)
    _rewrite(output, diagnostics)
    with pytest.raises(ValueError, match="query diagnostics exact/derived"):
        experiment.verify_artifacts(output)

    digest = copy.deepcopy(focused_report)
    digest["training"][0]["final_state_sha256"] = "0" * 64
    experiment.write_artifacts(focused_report, output)
    _rewrite(output, digest)
    with pytest.raises(ValueError, match="final state digest"):
        experiment.verify_artifacts(output)


def test_verifier_recomputes_provenance_and_markdown(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output = tmp_path / "radio-spectrum.json"
    wrong_source = copy.deepcopy(focused_report)
    wrong_source["provenance"]["sources"][0]["sha256"] = "0" * 64
    experiment.write_artifacts(wrong_source, output)
    with pytest.raises(ValueError, match="provenance"):
        experiment.verify_artifacts(output)

    experiment.write_artifacts(focused_report, output)
    output.with_suffix(".md").write_text("unbound\n")
    with pytest.raises(ValueError, match="Markdown mismatch"):
        experiment.verify_artifacts(output)
