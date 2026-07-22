"""Focused contracts for the lease-state scaling experiment."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.performance.experiments import lease_state_scaling as experiment
from treemendous.applications._shared.leasing import LeaseState
from treemendous.domain import Span


def _synthetic_row(kind: str, count: int, blocks: int) -> dict[str, Any]:
    if kind == "no_expiry_acquire_release":
        candidate_ns = count * 100
        baseline_ns = count * 1_000
        cycles = experiment.WRITE_CYCLES
    else:
        candidate_ns = 100
        baseline_ns = 1_000
        cycles = 1 if kind == "expiry_burst" else experiment.READ_CYCLES
    raw_blocks = []
    candidate_totals = []
    baseline_totals = []
    for block_index in range(blocks):
        first = (
            ["candidate", "baseline"]
            if block_index % 2 == 0
            else ["baseline", "candidate"]
        )
        positions = [
            {
                "first": ordering,
                "candidate_ns": candidate_ns,
                "baseline_ns": baseline_ns,
            }
            for ordering in first
        ]
        candidate_total = candidate_ns * 2
        baseline_total = baseline_ns * 2
        raw_blocks.append(
            {
                "block_index": block_index,
                "positions": positions,
                "candidate_ns_total": candidate_total,
                "baseline_ns_total": baseline_total,
                "ratio": candidate_total / baseline_total,
            }
        )
        candidate_totals.append(candidate_total)
        baseline_totals.append(baseline_total)
    ratios = [
        candidate / baseline
        for candidate, baseline in zip(candidate_totals, baseline_totals, strict=True)
    ]
    seed = experiment.BOOTSTRAP_SEED + count * 17 + sum(map(ord, kind))
    divisor = float(2 * cycles)
    row = {
        "kind": kind,
        "active_leases": count,
        "cycles_per_position": cycles,
        "validated_blocks": blocks,
        "blocks": raw_blocks,
        "block_ratios": ratios,
        "candidate_ns_per_cycle": experiment._summary(
            [value / divisor for value in candidate_totals], seed + 1
        ),
        "baseline_ns_per_cycle": experiment._summary(
            [value / divisor for value in baseline_totals], seed + 2
        ),
        "ratio": experiment._summary(ratios, seed),
        "result_sha256": experiment._semantic_digests(kind, count)[0],
        "final_state_sha256": experiment._semantic_digests(kind, count)[1],
    }
    if kind == "no_expiry_acquire_release":
        row.update(
            {
                "candidate_active_lease_replays": 0,
                "baseline_active_lease_replays": count * blocks * 4,
            }
        )
    elif kind == "expiry_burst":
        row["expired_per_cycle"] = max(1, count // 8)
    elif kind == "fence_validation":
        row["baseline_linear_lease_visits"] = (
            count * blocks * 2 * experiment.READ_CYCLES
        )
    return row


def _synthetic_report() -> dict[str, Any]:
    blocks = experiment.SMOKE_BLOCKS
    rows = [
        _synthetic_row(kind, count, blocks)
        for count in experiment.ACTIVE_LEASES
        for kind in (
            "repeated_snapshot",
            "fence_validation",
            "no_expiry_acquire_release",
            "expiry_burst",
        )
    ]
    instrumentation = {"public_snapshot_calls": 0, "linear_lease_scans": 0}
    memory = {
        "restorative_cycles": 1_000,
        "bytes_after_100": 1_000,
        "bytes_after_1000": 1_000,
        "settling_ratio": 1.0,
        "limit": experiment.MEMORY_SETTLING_LIMIT,
    }
    return {
        "schema": experiment.SCHEMA,
        "blocks": blocks,
        "methodology": experiment._methodology(blocks),
        "instrumentation": instrumentation,
        "rows": rows,
        "memory": memory,
        "gate": experiment._gate(rows, instrumentation, memory),
        "provenance": experiment._provenance(),
    }


def test_experiment_only_candidate_clones_free_state_without_runtime_seam() -> None:
    pool, _, _ = experiment._seed_pool(32)
    replays = [0]
    experiment._observe_active_replays(pool, replays)
    target = Span(32, 33)

    acquired = experiment._candidate_acquire_exact(pool, "candidate", target)
    released = experiment._candidate_release(pool, acquired)

    assert acquired.resource == target
    assert released.state is LeaseState.RELEASED
    assert replays == [0]
    assert pool.snapshot().diagnostics.active_leases == 32
    assert not hasattr(pool, "_clone_free")


def _write(output: Path, report: dict[str, Any]) -> None:
    experiment.write_artifacts(report, output)


def test_verifier_rejects_matrix_schema_type_digest_and_derived_tampering(
    tmp_path: Path,
) -> None:
    report = _synthetic_report()
    output = tmp_path / "lease.json"

    duplicate = copy.deepcopy(report)
    duplicate["rows"].insert(1, copy.deepcopy(duplicate["rows"][0]))
    duplicate["rows"].pop()
    _write(output, duplicate)
    with pytest.raises(ValueError, match="ordered unique matrix"):
        experiment.verify_artifacts(output)

    missing = copy.deepcopy(report)
    del missing["rows"][0]["final_state_sha256"]
    _write(output, missing)
    with pytest.raises(ValueError, match="per-kind row schema"):
        experiment.verify_artifacts(output)

    relabeled = copy.deepcopy(report)
    relabeled["rows"][0]["kind"] = "fence_validation"
    _write(output, relabeled)
    with pytest.raises(ValueError, match="ordered unique matrix"):
        experiment.verify_artifacts(output)

    bool_int = copy.deepcopy(report)
    bool_int["rows"][0]["active_leases"] = True
    _write(output, bool_int)
    with pytest.raises(ValueError, match="identity exact type"):
        experiment.verify_artifacts(output)

    semantic = copy.deepcopy(report)
    semantic["rows"][0]["result_sha256"] = "0" * 64
    _write(output, semantic)
    with pytest.raises(ValueError, match="semantic digest"):
        experiment.verify_artifacts(output)

    instrumentation = copy.deepcopy(report)
    instrumentation["rows"][1]["baseline_linear_lease_visits"] += 1
    _write(output, instrumentation)
    with pytest.raises(ValueError, match="fence instrumentation"):
        experiment.verify_artifacts(output)

    root_instrumentation = copy.deepcopy(report)
    root_instrumentation["instrumentation"]["public_snapshot_calls"] = 1
    _write(output, root_instrumentation)
    with pytest.raises(ValueError, match="instrumentation derivation"):
        experiment.verify_artifacts(output)

    memory = copy.deepcopy(report)
    memory["memory"]["settling_ratio"] = 1.01
    _write(output, memory)
    with pytest.raises(ValueError, match="memory derivation"):
        experiment.verify_artifacts(output)


def test_verifier_rejects_duplicate_key_and_noncanonical_bytes(tmp_path: Path) -> None:
    report = _synthetic_report()
    output = tmp_path / "lease.json"
    _write(output, report)

    duplicated = output.read_text().replace(
        f'  "schema": "{experiment.SCHEMA}"',
        f'  "schema": "duplicate",\n  "schema": "{experiment.SCHEMA}"',
        1,
    )
    encoded = duplicated.encode()
    output.write_bytes(encoded)
    Path(f"{output}.sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {output.name}\n"
    )
    with pytest.raises(ValueError, match="duplicate key"):
        experiment.verify_artifacts(output)

    noncanonical = json.dumps(report, sort_keys=True).encode()
    output.write_bytes(noncanonical)
    Path(f"{output}.sha256").write_text(
        f"{hashlib.sha256(noncanonical).hexdigest()}  {output.name}\n"
    )
    with pytest.raises(ValueError, match="not canonical"):
        experiment.verify_artifacts(output)


def test_generation_then_separate_process_verify_is_byte_stable(
    tmp_path: Path,
) -> None:
    report = _synthetic_report()
    gate = report["gate"]
    assert gate["candidate_b"]["decision"] == "ACCEPTED"
    assert gate["candidate_d"]["decision"] == "REJECTED"
    assert gate["candidate_d"]["checks"] == {
        "no_expiry_zero_active_replays": True,
        "no_expiry_n8192_upper_at_most_0_50": True,
        "candidate_n8192_over_n2048_at_most_2": False,
        "write_cells_upper_at_most_1_10": True,
        "retained_memory_settles_within_10_percent": True,
    }

    output = tmp_path / "lease-state-scaling-smoke.json"
    paths = experiment.write_artifacts(report, output)
    before = tuple(path.read_bytes() for path in paths)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.performance.experiments.lease_state_scaling",
            "--verify",
            "--output",
            str(output),
        ],
        cwd=experiment._ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert tuple(path.read_bytes() for path in paths) == before
