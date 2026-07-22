"""Strict contracts for segmented ExactBatch storage qualification evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.performance.experiments import exact_batch_storage_matrix as matrix

ARCHIVE = matrix._REPOSITORY_ROOT / "docs/evidence/experiments" / matrix.ARCHIVE_NAME

# The 199 KB measured rejection artifact is reproducible evidence, not tracked
# source. Reproduce it with `just reproduce-exact-batch-storage-rejection` (or
# fetch the release asset) into docs/evidence/experiments to exercise the
# archive-data contracts below. The durability proof itself—tracked patch plus
# baseline/segmented source SHAs—runs unconditionally via
# ``test_patch_applies_to_baseline_and_produces_archived_source_hash``.
requires_archive = pytest.mark.skipif(
    not ARCHIVE.is_file(),
    reason="segmented rejection archive is reproducible/release evidence, not tracked",
)


@requires_archive
def test_failed_small_n_confirmation_records_segmented_runtime_rejection() -> None:
    report = matrix.verify_archive_artifacts(ARCHIVE)
    rejection: dict[str, Any] = report["archive"]["rejection"]
    assert rejection == {
        "decision": "REJECTED",
        "workload": "promotion-restorative-n64-b16",
        "balanced_blocks": 20,
        "candidate_over_vector_median": 1.0652732286470954,
        "candidate_over_vector_confidence_95": [
            1.0406022687634031,
            1.1159159624697532,
        ],
        "upper95_gate": 1.10,
    }
    assert (
        rejection["candidate_over_vector_confidence_95"][1] > rejection["upper95_gate"]
    )


def test_full_matrix_identity_covers_required_dimensions_and_block_boundaries() -> None:
    definitions = matrix.diagnostic_definitions()
    regular = [item for item in definitions if item["case_id"].startswith("matrix-")]
    blocks = [item for item in definitions if item["case_id"].startswith("block-")]

    assert len(regular) == 4 * 4 * 9
    assert {item["interval_count"] for item in regular} == {64, 1_000, 10_000, 100_000}
    assert {item["batch_size"] for item in regular} == {0, 1, 16, 256}
    assert {item["shape"] for item in regular} == set(matrix.DIAGNOSTIC_SHAPES)
    block_boundaries = {(item["interval_count"], item["shape"]) for item in blocks}
    expected_boundaries = {
        (count, f"block_{shape}")
        for count in (matrix.K - 1, matrix.K, matrix.K + 1)
        for shape in matrix.BLOCK_SHAPES
    }
    assert block_boundaries == expected_boundaries


def test_promotion_matrix_has_every_fixed_gate_cell_and_separate_layers() -> None:
    definitions = matrix.promotion_definitions()
    ids = {item["case_id"] for item in definitions}
    assert "promotion-restorative-n64-b16" in ids
    assert "promotion-local-n100000-b16" in ids
    wide_ids = {f"promotion-wide-n100000-p{value}" for value in (1, 10, 100)}
    snapshot_ids = {f"promotion-snapshot-n{count}" for count in matrix.INTERVAL_COUNTS}
    assert wide_ids <= ids
    assert snapshot_ids <= ids
    assert {item["layer"] for item in definitions} == {
        "construction",
        "mutate",
        "snapshot",
        "materialization",
    }


@requires_archive
def test_archive_workers_record_distinct_roots_and_native_binaries() -> None:
    report = matrix.verify_archive_artifacts(ARCHIVE)
    block = report["balanced_blocks"][0]
    baseline = block["baseline"]
    candidate = block["candidate"]

    assert Path(baseline["imports"]["treemendous"]["path"]).is_relative_to(
        Path(baseline["root"])
    )
    assert Path(candidate["imports"]["treemendous"]["path"]).is_relative_to(
        Path(candidate["root"])
    )
    assert baseline["root"] != candidate["root"]
    assert baseline["binary"]["path"] != candidate["binary"]["path"]
    assert baseline["binary"]["sha256"] != candidate["binary"]["sha256"]


def test_worker_rejects_when_both_roots_resolve_to_candidate() -> None:
    candidate_root = matrix._REPOSITORY_ROOT.resolve()
    with pytest.raises(ValueError, match="same native binary path"):
        matrix._run_worker(
            candidate_root,
            "promotion",
            "smoke",
            comparison_root=candidate_root,
        )


@pytest.fixture(scope="module")
def smoke_artifact() -> tuple[Path, dict[str, Any]]:
    if not ARCHIVE.is_file():
        pytest.skip(
            "segmented rejection archive is reproducible/release evidence, not tracked"
        )
    return ARCHIVE, matrix.verify_archive_artifacts(ARCHIVE)


def _rewrite(output: Path, report: dict[str, Any]) -> None:
    matrix.write_artifacts(report, output)


def test_smoke_artifact_round_trips_strict_archive_verifier(
    smoke_artifact: tuple[Path, dict[str, Any]],
) -> None:
    output, report = smoke_artifact
    verified = matrix.verify_archive_artifacts(output)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.performance.experiments.exact_batch_storage_matrix",
            "--verify",
            "--archive",
            "--output",
            str(output),
        ],
        cwd=matrix._REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert verified["archive"]["rejection"]["decision"] == "REJECTED"
    assert len(verified["balanced_blocks"]) == matrix.PROMOTION_BLOCKS
    assert verified["promotion_rows"] == report["promotion_rows"]


def test_patch_applies_to_baseline_and_produces_archived_source_hash() -> None:
    matrix._verify_patch_application()


def test_verifier_rejects_matrix_and_derived_tampering(
    tmp_path: Path, smoke_artifact: tuple[Path, dict[str, Any]]
) -> None:
    _, report = smoke_artifact
    output = tmp_path / matrix.ARCHIVE_NAME

    identity = copy.deepcopy(report)
    identity["matrix_identity"]["batch_sizes"] = [16]
    _rewrite(output, identity)
    with pytest.raises(ValueError, match="matrix identity"):
        matrix.verify_archive_artifacts(output)

    derived = copy.deepcopy(report)
    derived["promotion_rows"][0]["candidate_over_vector_median"] += 0.25
    _rewrite(output, derived)
    with pytest.raises(ValueError, match="promotion raw derivation"):
        matrix.verify_archive_artifacts(output)

    duplicate = copy.deepcopy(report)
    duplicate["balanced_blocks"][0]["candidate"]["cells"].insert(
        1, copy.deepcopy(duplicate["balanced_blocks"][0]["candidate"]["cells"][0])
    )
    duplicate["balanced_blocks"][0]["candidate"]["cells"].pop()
    _rewrite(output, duplicate)
    with pytest.raises(ValueError, match="matrix order/uniqueness"):
        matrix.verify_archive_artifacts(output)

    oracle = copy.deepcopy(report)
    oracle["balanced_blocks"][0]["candidate"]["cells"][0]["oracle_digest"] = "0" * 64
    _rewrite(output, oracle)
    with pytest.raises(ValueError, match="cell derivation"):
        matrix.verify_archive_artifacts(output)


def test_verifier_rejects_duplicate_keys_nonfinite_and_provenance_tampering(
    tmp_path: Path, smoke_artifact: tuple[Path, dict[str, Any]]
) -> None:
    _, report = smoke_artifact
    output = tmp_path / matrix.ARCHIVE_NAME

    provenance = copy.deepcopy(report)
    provenance["provenance"]["candidate"]["binary"]["sha256"] = "0" * 64
    _rewrite(output, provenance)
    with pytest.raises(ValueError, match="provenance|binary"):
        matrix.verify_archive_artifacts(output)

    patch = copy.deepcopy(report)
    patch["archive"]["patch"]["sha256"] = "0" * 64
    _rewrite(output, patch)
    with pytest.raises(ValueError, match="patch/rejection binding"):
        matrix.verify_archive_artifacts(output)

    _rewrite(output, report)
    text = output.read_text().replace(
        f'  "schema": "{matrix.SCHEMA}"',
        f'  "schema": "duplicate",\n  "schema": "{matrix.SCHEMA}"',
        1,
    )
    encoded = text.encode()
    output.write_bytes(encoded)
    Path(f"{output}.sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {output.name}\n"
    )
    with pytest.raises(ValueError, match="duplicate key"):
        matrix.verify_archive_artifacts(output)

    nonfinite = copy.deepcopy(report)
    nonfinite["promotion_rows"][0]["candidate_over_vector_median"] = math.nan
    encoded = (json.dumps(nonfinite, indent=2, sort_keys=True) + "\n").encode()
    output.write_bytes(encoded)
    Path(f"{output}.sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {output.name}\n"
    )
    with pytest.raises(ValueError, match="non-finite"):
        matrix.verify_archive_artifacts(output)
