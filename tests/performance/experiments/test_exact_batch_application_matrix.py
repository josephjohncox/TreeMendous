"""Focused contracts for the bounded exact-batch application diagnostic."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.performance.experiments import exact_batch_application_matrix as matrix


def test_local_matrix_covers_fixed_states_batches_localities_and_shapes() -> None:
    cases = matrix._cases(
        matrix.LOCAL_INTERVAL_COUNTS,
        matrix.BATCH_SIZES,
        matrix.SHAPES,
        matrix.LOCALITIES,
    )
    assert len(cases) == 3 * 4 * 5 * 3
    assert {case.interval_count for case in cases} == {64, 1_000, 10_000}
    assert {case.batch_size for case in cases} == {1, 4, 16, 64}
    assert {case.locality for case in cases} == {"head", "middle", "tail"}
    assert {case.shape for case in cases} == set(matrix.SHAPES)
    assert matrix.case_definition(100_000, 16, "wide_fanout", "tail").fanout == 8


@pytest.mark.parametrize("shape", matrix.SHAPES)
@pytest.mark.parametrize("locality", matrix.LOCALITIES)
@pytest.mark.parametrize("batch_size", (1, 4))
def test_generated_application_shapes_match_cpp_boundary(
    shape: str, locality: str, batch_size: int
) -> None:
    case = matrix.case_definition(64, batch_size, shape, locality)
    rows, final_geometry, packed_bytes, work = matrix.attest_case(case)
    assert len(rows) == batch_size
    assert final_geometry
    assert packed_bytes > 0
    assert work["logical_rows"] == batch_size
    assert work["initial_live_intervals"] == 64


def test_shape_declarations_capture_rejections_noops_restore_and_fanout() -> None:
    strict = matrix._oracle_evidence(
        matrix.case_definition(64, 4, "strict_accept_reject", "middle")
    )[2]
    assert strict["strict_accepted_rows"] == 1
    assert strict["strict_rejected_rows"] == 1

    idempotent = matrix._oracle_evidence(
        matrix.case_definition(64, 4, "idempotent_real_noop", "middle")
    )[2]
    assert idempotent["mutating_rows"] == 2
    assert idempotent["noop_rows"] == 2
    assert idempotent["initial_live_intervals"] == idempotent["final_live_intervals"]

    for shape in ("fragment_restore", "coalesce_restore"):
        work = matrix._oracle_evidence(matrix.case_definition(64, 4, shape, "middle"))[
            2
        ]
        assert work["mutating_rows"] == 4
        assert work["initial_live_intervals"] == work["final_live_intervals"]

    wide_case = matrix.case_definition(64, 1, "wide_fanout", "middle")
    wide_rows, _, wide_work = matrix._oracle_evidence(wide_case)
    assert len(wide_rows[0].changed) == 8
    assert wide_work["fanout"] == 8


@pytest.fixture(scope="module")
def focused_report() -> dict[str, Any]:
    return matrix.run_benchmark(profile="smoke", samples=10)


def _write_tampered_report(
    tmp_path: Path, report: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    output = tmp_path / "matrix.json"
    tampered = copy.deepcopy(report)
    matrix.write_artifacts(tampered, output)
    return output, tampered


def _refresh_workload_digests(report: dict[str, Any]) -> None:
    manifest = report["workload_manifest"]
    case = manifest["cases"][0]
    case["digest"] = matrix._checksum(
        {key: value for key, value in case.items() if key != "digest"}
    )
    manifest["digest"] = matrix._checksum(
        {key: value for key, value in manifest.items() if key != "digest"}
    )


def _rewrite_json_and_checksum(output: Path, text: str) -> None:
    encoded = text.encode()
    output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    Path(f"{output}.sha256").write_text(f"{digest}  {output.name}\n")


def test_small_matrix_writes_and_verifies_canonical_triplet(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output = tmp_path / "matrix.json"
    json_path, markdown_path, checksum_path = matrix.write_artifacts(
        focused_report, output
    )
    verified = matrix.verify_artifacts(output)
    before = tuple(
        path.read_bytes() for path in (json_path, markdown_path, checksum_path)
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.performance.experiments.exact_batch_application_matrix",
            "--verify",
            "--output",
            str(output),
        ],
        cwd=matrix._REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert verified["schema"] == matrix.SCHEMA
    assert len(verified["rows"][0]["exact_latency_ns_samples"]) == 10
    assert all(path.is_file() for path in (json_path, markdown_path, checksum_path))
    assert (
        tuple(path.read_bytes() for path in (json_path, markdown_path, checksum_path))
        == before
    )


def test_verify_rejects_duplicate_json_keys(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output, _ = _write_tampered_report(tmp_path, focused_report)
    text = output.read_text().replace(
        f'  "schema": "{matrix.SCHEMA}"',
        f'  "schema": "duplicate",\n  "schema": "{matrix.SCHEMA}"',
        1,
    )
    _rewrite_json_and_checksum(output, text)

    with pytest.raises(ValueError, match="duplicate key"):
        matrix.verify_artifacts(output)


@pytest.mark.parametrize(
    ("constant", "value"),
    (("NaN", float("nan")), ("Infinity", float("inf")), ("-Infinity", -float("inf"))),
)
def test_verify_rejects_non_finite_json_numbers(
    tmp_path: Path,
    focused_report: dict[str, Any],
    constant: str,
    value: float,
) -> None:
    output, tampered = _write_tampered_report(tmp_path, focused_report)
    tampered["rows"][0]["exact_latency_ns_samples"][0] = value
    text = json.dumps(tampered, indent=2, sort_keys=True) + "\n"
    assert constant in text
    _rewrite_json_and_checksum(output, text)

    with pytest.raises(ValueError, match="non-finite number"):
        matrix.verify_artifacts(output)


def test_verify_rejects_finite_syntax_that_overflows_json_float(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output, tampered = _write_tampered_report(tmp_path, focused_report)
    tampered["rows"][0]["exact_latency_ns_samples"][0] = float("inf")
    text = (json.dumps(tampered, indent=2, sort_keys=True) + "\n").replace(
        "Infinity", "1e999", 1
    )
    _rewrite_json_and_checksum(output, text)

    with pytest.raises(ValueError, match="non-finite number"):
        matrix.verify_artifacts(output)


@pytest.mark.parametrize(
    "replacement",
    (pytest.param(True, id="bool-int"), pytest.param(1.0, id="int-float")),
)
def test_verify_rejects_json_numeric_type_coercion_in_reconstructed_manifest(
    tmp_path: Path,
    focused_report: dict[str, Any],
    replacement: bool | float,
) -> None:
    output, tampered = _write_tampered_report(tmp_path, focused_report)
    tampered["workload_manifest"]["cases"][0]["work"]["fanout"] = replacement
    _refresh_workload_digests(tampered)
    matrix.write_artifacts(tampered, output)

    with pytest.raises(ValueError, match="reconstructed matrix"):
        matrix.verify_artifacts(output)


def test_verify_rejects_checksum_tamper(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output, _ = _write_tampered_report(tmp_path, focused_report)
    Path(f"{output}.sha256").write_text(f"{'0' * 64}  {output.name}\n")

    with pytest.raises(ValueError, match="checksum"):
        matrix.verify_artifacts(output)


def test_verify_rejects_markdown_tamper(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output, _ = _write_tampered_report(tmp_path, focused_report)
    output.with_suffix(".md").write_text("tampered\n")

    with pytest.raises(ValueError, match="Markdown"):
        matrix.verify_artifacts(output)


def test_verify_rejects_raw_sample_tamper(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output, tampered = _write_tampered_report(tmp_path, focused_report)
    tampered["rows"][0]["exact_latency_ns_samples"][0] += 1
    matrix.write_artifacts(tampered, output)

    with pytest.raises(ValueError, match="raw samples"):
        matrix.verify_artifacts(output)


def test_verify_rejects_derived_value_tamper(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output, tampered = _write_tampered_report(tmp_path, focused_report)
    tampered["rows"][0]["paired_exact_over_scalar_median"] += 1.0
    matrix.write_artifacts(tampered, output)

    with pytest.raises(ValueError, match="paired_exact_over_scalar_median"):
        matrix.verify_artifacts(output)


def test_verify_rejects_workload_digest_tamper(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output, tampered = _write_tampered_report(tmp_path, focused_report)
    tampered["workload_manifest"]["digest"] = "0" * 64
    matrix.write_artifacts(tampered, output)

    with pytest.raises(ValueError, match="workload digest"):
        matrix.verify_artifacts(output)


def test_git_metadata_binds_untracked_sources_and_reports_staged_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.py"
    tracked.write_text("VALUE = 1\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=tmp_path,
        check=True,
    )
    experiment = tmp_path / "tests/performance/experiments/new.py"
    example = tmp_path / "examples/patterns/new.py"
    experiment.parent.mkdir(parents=True)
    example.parent.mkdir(parents=True)
    experiment.write_text("EXPERIMENT = 1\n")
    example.write_text("EXAMPLE = 1\n")
    tracked.write_text("VALUE = 2\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    monkeypatch.setattr(matrix, "_REPOSITORY_ROOT", tmp_path)

    first = matrix._git_metadata()
    assert first["staged_files"] == ["tracked.py"]
    assert "tests/performance/experiments/new.py" in first["changed_paths"]
    assert "examples/patterns/new.py" in first["changed_paths"]

    experiment.write_text("EXPERIMENT = 2\n")
    second = matrix._git_metadata()
    assert first["source_state_sha256"] != second["source_state_sha256"]


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
        ("backends", "exact_batch", "id"),
        ("backends", "exact_batch", "module"),
        ("backends", "exact_batch", "type"),
        ("backends", "exact_batch", "extension", "path"),
        ("backends", "exact_batch", "extension", "sha256"),
    ),
)
def test_verify_rejects_each_runtime_build_and_backend_provenance_category(
    tmp_path: Path, focused_report: dict[str, Any], path: tuple[str, ...]
) -> None:
    output, tampered = _write_tampered_report(tmp_path, focused_report)
    target = tampered["provenance"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = True
    matrix.write_artifacts(tampered, output)

    with pytest.raises(ValueError, match="runtime|build|backend"):
        matrix.verify_artifacts(output)


def test_verify_rejects_packed_bytes_row_shape_method_and_provenance_tamper(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output = tmp_path / "matrix.json"

    packed = copy.deepcopy(focused_report)
    packed["rows"][0]["packed_result_bytes"] += 1
    matrix.write_artifacts(packed, output)
    with pytest.raises(ValueError, match="reconstructed workload"):
        matrix.verify_artifacts(output)

    extra = copy.deepcopy(focused_report)
    extra["rows"][0]["extra"] = True
    matrix.write_artifacts(extra, output)
    with pytest.raises(ValueError, match="row keys/type"):
        matrix.verify_artifacts(output)

    missing = copy.deepcopy(focused_report)
    del missing["rows"][0]["packed_result_bytes"]
    text = json.dumps(missing, indent=2, sort_keys=True) + "\n"
    _rewrite_json_and_checksum(output, text)
    output.with_suffix(".md").write_text("unused\n")
    with pytest.raises(ValueError, match="row keys/type"):
        matrix.verify_artifacts(output)

    method = copy.deepcopy(focused_report)
    method["methodology"]["pair_order"] = "tampered"
    matrix.write_artifacts(method, output)
    with pytest.raises(ValueError, match="fixed methodology"):
        matrix.verify_artifacts(output)

    provenance = copy.deepcopy(focused_report)
    provenance["provenance"]["sources"][0]["sha256"] = "0" * 64
    matrix.write_artifacts(provenance, output)
    with pytest.raises(ValueError, match="source file path/hash"):
        matrix.verify_artifacts(output)

    extension = copy.deepcopy(focused_report)
    extension["provenance"]["backends"]["exact_batch"]["extension"]["sha256"] = "0" * 64
    matrix.write_artifacts(extension, output)
    with pytest.raises(ValueError, match="backend/extension provenance"):
        matrix.verify_artifacts(output)
