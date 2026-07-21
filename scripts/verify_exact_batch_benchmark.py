#!/usr/bin/env python3
"""Strictly verify exact-batch benchmark and scoped scalar evidence artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from scripts.verify_mutation_attribution import verify_artifact as verify_attribution
from tests.performance.mutation_attribution import PRIMARY_LAYER

SCHEMA = "treemendous-experimental-exact-batch-benchmark-v2"
WORKLOAD_SCHEMA = "treemendous-exact-batch-restorative-workload-v1"
BASELINE_BACKEND = "cpp_boundary"
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
SAMPLES_MINIMUM = 20
BOOTSTRAP_RESAMPLES = 10_000
PAIRED_BOOTSTRAP_SEED = 50
THROUGHPUT_BOOTSTRAP_SEED = 16
SCALAR_REGRESSION_LIMIT = 1.03
EXPECTED_THRESHOLDS = {
    "batch16_ops_per_second_lower_95": 2_000_000.0,
    "batch16_speedup_lower_95": 2.0,
    "batch4_speedup_lower_95": 1.0,
    "stable_scalar_candidate_over_baseline_upper_95": SCALAR_REGRESSION_LIMIT,
}
_EXTENSION_MODULES = {
    "exact_batch": "treemendous.cpp._exact_batch",
    BASELINE_BACKEND: "treemendous.cpp.boundary",
}
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _strict_json(encoded: bytes) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value is not permitted: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is not permitted: {key}")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            encoded,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("exact-batch JSON is invalid") from exc
    if not isinstance(decoded, dict):
        raise ValueError("exact-batch JSON root must be an object")
    return decoded


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def _boolean(value: Any, description: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{description} must be a boolean")
    return value


def _exact_fields(value: dict[str, Any], expected: set[str], description: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{description} fields are incomplete or unexpected")


def _valid_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _positive_integer_samples(
    value: Any, description: str, *, expected_count: int
) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != expected_count
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item <= 0
            or item > 2**63 - 1
            for item in value
        )
    ):
        raise ValueError(
            f"{description} must contain exactly {expected_count} positive "
            "signed 64-bit integer samples"
        )
    return value


def _checksum(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _median_confidence(values: list[float], seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    bootstrap = [
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return _percentile(bootstrap, 0.025), _percentile(bootstrap, 0.975)


def _paired_statistics(baseline: list[int], candidate: list[int]) -> dict[str, Any]:
    ratios = [right / left for left, right in zip(baseline, candidate, strict=True)]
    confidence = _median_confidence(ratios, PAIRED_BOOTSTRAP_SEED)
    median_ratio = statistics.median(ratios)
    if median_ratio > 1.0:
        classification = "fail"
    elif confidence[1] <= 1.0:
        classification = "pass"
    else:
        classification = "inconclusive"
    return {
        "sample_count": len(ratios),
        "baseline_samples_ns": baseline,
        "candidate_samples_ns": candidate,
        "paired_ratios": ratios,
        "median_ratio": median_ratio,
        "median_improvement": 1.0 - median_ratio,
        "confidence_95_ratio": list(confidence),
        "ratio_limit": 1.0,
        "classification": classification,
    }


def _multi_component_block(start_index: int) -> list[list[int]]:
    base = start_index * 16
    return [
        [0, base, base + 56],
        [1, base + 8, base + 16],
        [1, base + 24, base + 32],
        [1, base + 40, base + 48],
    ]


def _trace64() -> list[list[int]]:
    blocks = [
        [[1, 2, 6], [0, 2, 6], [2, 8, 12], [1, 8, 12]],
        _multi_component_block(0),
        [[1, 132, 140], [0, 132, 136], [0, 128, 136], [0, 128, 136]],
        [[2, 264, 268], [2, 264, 268], [1, 258, 262], [0, 258, 262]],
    ]
    for block_index in range(4, 16):
        if block_index % 2:
            blocks.append(_multi_component_block(((block_index * 7) % 13) * 4))
        else:
            interval_index = (block_index * 11) % 64
            base = interval_index * 16
            blocks.append(
                [
                    [1, base + 1, base + 7],
                    [0, base + 1, base + 7],
                    [0, base, base + 8],
                    [2, base + 8, base + 13],
                ]
            )
    return [operation for block in blocks for operation in block]


def _expected_manifest() -> dict[str, Any]:
    trace64 = _trace64()
    traces: dict[str, Any] = {}
    for size in BATCH_SIZES:
        trace = {
            "classification": (
                "single-call-no-op-diagnostic" if size == 1 else "restorative"
            ),
            "operations": [[0, 0, 8]] if size == 1 else trace64[:size],
        }
        traces[str(size)] = {**trace, "digest": _checksum(trace)}
    body = {
        "schema": WORKLOAD_SCHEMA,
        "domain": [0, 1_024],
        "initial_intervals": [[index * 16, index * 16 + 8] for index in range(64)],
        "traces": traces,
    }
    return {**body, "digest": _checksum(body)}


def _render_markdown(report: dict[str, Any], digest: str) -> str:
    lines = [
        "# Experimental exact-batch benchmark",
        "",
        f"- Candidate: `{report['candidate']['commit']}`",
        f"- Clean worktree: `{str(report['candidate']['clean_worktree']).lower()}`",
        f"- Baseline: `{report['baseline_backend']}`",
        f"- Workload digest: `{report['workload_manifest']['digest']}`",
        f"- JSON SHA-256: `{digest}`",
        "",
        "| Gate | Observed lower 95% bound | Threshold | Result |",
        "|---|---:|---:|---:|",
    ]
    for name in ("break_even_by_4", "batch16_speedup", "batch16_absolute"):
        gate = report["gates"][name]
        lines.append(
            f"| {name} | {gate['observed']:.6g} | {gate['threshold']:.6g} | "
            f"{'pass' if gate['passed'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "> Stable scalar regression is separate quick attribution evidence; "
            "it is not a full scalar-promotion claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def _loaded_extension_metadata(module_name: str) -> dict[str, str]:
    module = importlib.import_module(module_name)
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str):
        raise ValueError(f"loaded native extension {module_name!r} has no file")
    try:
        path = Path(raw_path).resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"loaded native extension {module_name!r} file is unavailable"
        ) from exc
    try:
        display_path = str(path.relative_to(_REPOSITORY_ROOT))
    except ValueError:
        display_path = str(path)
    return {
        "path": display_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _verify_metadata(report: dict[str, Any], expected_candidate: str | None) -> str:
    candidate = _mapping(report.get("candidate"), "candidate")
    _exact_fields(candidate, {"commit", "clean_worktree"}, "candidate")
    commit = candidate.get("commit")
    if not _valid_commit(commit):
        raise ValueError("candidate commit is invalid")
    if expected_candidate is not None and commit != expected_candidate:
        raise ValueError("candidate commit does not match")
    if not _boolean(candidate.get("clean_worktree"), "candidate cleanliness"):
        raise ValueError("exact-batch artifact was produced from a dirty worktree")

    environment = _mapping(report.get("environment"), "environment")
    _exact_fields(
        environment,
        {
            "python",
            "implementation",
            "python_compiler",
            "platform",
            "machine",
            "processor",
            "cpu_count",
        },
        "environment",
    )
    if any(
        not isinstance(environment[field], str) or not environment[field]
        for field in environment
        if field != "cpu_count"
    ) or (
        isinstance(environment["cpu_count"], bool)
        or not isinstance(environment["cpu_count"], int)
        or environment["cpu_count"] < 0
    ):
        raise ValueError("environment metadata is invalid")

    build = _mapping(report.get("build"), "build")
    _exact_fields(
        build,
        {"command", "cxx", "cc", "cflags", "build_flags", "extensions"},
        "build",
    )
    for field in ("command", "cxx", "cc", "cflags"):
        if not isinstance(build[field], str) or not build[field]:
            raise ValueError("compiler/build metadata is invalid")
    flags = _mapping(build["build_flags"], "build flags")
    expected_flags = {
        "TREE_MENDOUS_DISABLE_OPTIMIZATIONS",
        "TREE_MENDOUS_LOCAL_NATIVE",
        "TREE_MENDOUS_SANITIZERS",
        "TREE_MENDOUS_GLIBCXX_DEBUG",
    }
    _exact_fields(flags, expected_flags, "build flags")
    if any(not isinstance(value, str) for value in flags.values()):
        raise ValueError("build flags are invalid")
    extensions = _mapping(build["extensions"], "extensions")
    _exact_fields(extensions, {"exact_batch", BASELINE_BACKEND}, "extensions")
    for name, raw_extension in extensions.items():
        extension = _mapping(raw_extension, f"{name} extension")
        _exact_fields(extension, {"path", "sha256"}, f"{name} extension")
        if not isinstance(extension["path"], str) or not extension["path"]:
            raise ValueError(f"{name} extension path is invalid")
        if not _valid_sha256(extension["sha256"]):
            raise ValueError(f"{name} extension digest is invalid")
        loaded = _loaded_extension_metadata(_EXTENSION_MODULES[name])
        if extension["path"] != loaded["path"]:
            raise ValueError(f"{name} extension path does not match loaded module")
        if extension["sha256"] != loaded["sha256"]:
            raise ValueError(f"{name} extension digest does not match loaded module")
    return str(commit)


def _verify_methodology(
    report: dict[str, Any], require_samples: int | None
) -> tuple[int, int]:
    methodology = _mapping(report.get("methodology"), "methodology")
    _exact_fields(
        methodology,
        {
            "samples",
            "target_operations",
            "paired_order",
            "paired_bootstrap_seed",
            "throughput_bootstrap_seed",
            "bootstrap_resamples",
            "initial_and_final_interval_count",
            "batch_sizes",
            "workload",
            "batch_1",
            "timed_batch_layer",
            "excluded",
        },
        "methodology",
    )
    samples = methodology.get("samples")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 20:
        raise ValueError("benchmark sample count is invalid")
    if require_samples is not None and samples != require_samples:
        raise ValueError("benchmark sample count does not match the requirement")
    fixed = {
        "paired_order": "alternating within each paired sample",
        "paired_bootstrap_seed": PAIRED_BOOTSTRAP_SEED,
        "throughput_bootstrap_seed": THROUGHPUT_BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "initial_and_final_interval_count": 64,
        "batch_sizes": list(BATCH_SIZES),
        "workload": "deterministic per-size restorative traces; four-operation blocks preserve the exact initial state",
        "batch_1": "separately labelled no-op call-overhead diagnostic",
        "timed_batch_layer": "buffer acquisition/copy, state staging, ordered execution, packed allocation, atomic commit, packed-result destruction",
        "excluded": "manager/setup construction, invariant snapshots, validation, and materialize",
    }
    if any(methodology.get(key) != value for key, value in fixed.items()):
        raise ValueError("benchmark methodology is inconsistent")
    target = methodology.get("target_operations")
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError("target operation count is invalid")
    return samples, target


def _verify_rows(
    report: dict[str, Any], samples: int, target_operations: int
) -> dict[str, dict[str, Any]]:
    rows = _mapping(report.get("rows"), "rows")
    if set(rows) != {str(size) for size in BATCH_SIZES}:
        raise ValueError("benchmark rows are incomplete")
    result: dict[str, dict[str, Any]] = {}
    row_fields = {
        "batch_size",
        "classification",
        "baseline_backend",
        "logical_operations_per_sample",
        "batch_ns_per_operation_samples",
        "scalar_ns_per_operation_samples",
        "batch_median_ops_per_second",
        "batch_ops_per_second_confidence_95",
        "paired",
        "speedup_confidence_95",
    }
    for size in BATCH_SIZES:
        description = f"batch-{size} row"
        row = _mapping(rows[str(size)], description)
        _exact_fields(row, row_fields, description)
        classification = "single-call-no-op-diagnostic" if size == 1 else "restorative"
        logical = row.get("logical_operations_per_sample")
        expected_logical = max(20, target_operations // size) * size
        if (
            row.get("batch_size") != size
            or row.get("classification") != classification
            or row.get("baseline_backend") != BASELINE_BACKEND
            or isinstance(logical, bool)
            or not isinstance(logical, int)
            or logical != expected_logical
        ):
            raise ValueError(f"{description} declaration is invalid")
        batch = _positive_integer_samples(
            row.get("batch_ns_per_operation_samples"),
            f"{description} batch samples",
            expected_count=samples,
        )
        scalar = _positive_integer_samples(
            row.get("scalar_ns_per_operation_samples"),
            f"{description} scalar samples",
            expected_count=samples,
        )
        expected_paired = _paired_statistics(scalar, batch)
        if row.get("paired") != expected_paired:
            raise ValueError(f"{description} paired derivations are inconsistent")
        throughput = [1_000_000_000 / value for value in batch]
        throughput_ci = list(_median_confidence(throughput, THROUGHPUT_BOOTSTRAP_SEED))
        if row.get("batch_median_ops_per_second") != statistics.median(throughput):
            raise ValueError(f"{description} throughput median is inconsistent")
        if row.get("batch_ops_per_second_confidence_95") != throughput_ci:
            raise ValueError(f"{description} throughput CI is inconsistent")
        ratio_ci = expected_paired["confidence_95_ratio"]
        speedup_ci = [1.0 / ratio_ci[1], 1.0 / ratio_ci[0]]
        if row.get("speedup_confidence_95") != speedup_ci:
            raise ValueError(f"{description} speedup CI is inconsistent")
        result[str(size)] = row
    return result


def _verify_gates(report: dict[str, Any], rows: dict[str, dict[str, Any]]) -> None:
    if report.get("thresholds") != EXPECTED_THRESHOLDS:
        raise ValueError("fixed thresholds are missing or inconsistent")
    expected_observed = {
        "batch16_absolute": rows["16"]["batch_ops_per_second_confidence_95"][0],
        "batch16_speedup": rows["16"]["speedup_confidence_95"][0],
        "break_even_by_4": rows["4"]["speedup_confidence_95"][0],
    }
    expected_threshold = {
        "batch16_absolute": EXPECTED_THRESHOLDS["batch16_ops_per_second_lower_95"],
        "batch16_speedup": EXPECTED_THRESHOLDS["batch16_speedup_lower_95"],
        "break_even_by_4": EXPECTED_THRESHOLDS["batch4_speedup_lower_95"],
    }
    gates = _mapping(report.get("gates"), "gates")
    if set(gates) != set(expected_observed):
        raise ValueError("exact-batch gates are incomplete")
    for name, observed in expected_observed.items():
        gate = _mapping(gates[name], f"{name} gate")
        _exact_fields(gate, {"threshold", "observed", "passed"}, f"{name} gate")
        threshold = expected_threshold[name]
        expected = {
            "threshold": threshold,
            "observed": observed,
            "passed": observed >= threshold,
        }
        if gate != expected:
            raise ValueError(f"{name} gate derivation is inconsistent")


def verify_report(
    report: dict[str, Any],
    *,
    expected_candidate: str | None = None,
    require_samples: int | None = None,
    enforce_gates: bool = False,
) -> dict[str, Any]:
    """Verify the exact-batch schema and every stored derivation."""
    _exact_fields(
        report,
        {
            "schema",
            "candidate",
            "environment",
            "build",
            "baseline_backend",
            "methodology",
            "workload_manifest",
            "rows",
            "materialize_16_ns_per_operation_samples",
            "thresholds",
            "gates",
        },
        "report",
    )
    if report.get("schema") != SCHEMA:
        raise ValueError("unexpected exact-batch schema")
    if report.get("baseline_backend") != BASELINE_BACKEND:
        raise ValueError("exact-batch baseline must be cpp_boundary")
    commit = _verify_metadata(report, expected_candidate)
    samples, target_operations = _verify_methodology(report, require_samples)
    if report.get("workload_manifest") != _expected_manifest():
        raise ValueError("restorative workload manifest or digest is inconsistent")
    rows = _verify_rows(report, samples, target_operations)
    _positive_integer_samples(
        report.get("materialize_16_ns_per_operation_samples"),
        "materialize samples",
        expected_count=samples,
    )
    _verify_gates(report, rows)
    if enforce_gates and not all(gate["passed"] for gate in report["gates"].values()):
        raise ValueError("one or more fixed exact-batch gates failed")
    return {"candidate_commit": commit, "samples": samples}


def verify_scalar_attribution(
    path: Path,
    *,
    expected_candidate: str,
    expected_baseline: str | None = None,
    require_samples: int | None = None,
) -> dict[str, Any]:
    """Verify quick attribution and independently apply only the scalar gate."""
    report = verify_attribution(
        path,
        expected_baseline=expected_baseline,
        expected_candidate=expected_candidate,
        require_samples=require_samples,
    )
    methodology = _mapping(report.get("methodology"), "scalar methodology")
    if _boolean(
        methodology.get("full_representative_suite"),
        "scalar representative-suite flag",
    ):
        raise ValueError("scalar evidence must be bounded quick/layers attribution")
    if not _boolean(
        report.get("semantic_checksums_match"), "scalar semantic consistency"
    ):
        raise ValueError("scalar semantic evidence is inconsistent")
    if not _boolean(report.get("environments_match"), "scalar environment match"):
        raise ValueError("scalar comparison environments do not match")
    if not _boolean(
        report.get("binary_provenance_complete"), "scalar binary provenance"
    ):
        raise ValueError("scalar native-binary provenance is incomplete")
    if not _boolean(report.get("clean_worktrees"), "scalar worktree cleanliness"):
        raise ValueError("scalar comparison worktrees are not clean")
    if report.get("status") != "diagnostic" or _boolean(
        report.get("promotion_eligible"), "scalar promotion eligibility"
    ):
        raise ValueError("quick scalar attribution validity status is inconsistent")
    baseline = _mapping(report.get("baseline"), "scalar baseline")
    candidate = _mapping(report.get("candidate"), "scalar candidate")
    baseline_commit = baseline.get("commit")
    candidate_commit = candidate.get("commit")
    if not _valid_commit(baseline_commit) or not _valid_commit(candidate_commit):
        raise ValueError("scalar baseline/candidate commits are invalid")
    if baseline_commit == candidate_commit:
        raise ValueError("scalar baseline and candidate commits must differ")
    if baseline.get("dirty") != "false" or candidate.get("dirty") != "false":
        raise ValueError("scalar baseline/candidate worktrees must be clean")
    if expected_baseline is not None and baseline_commit != expected_baseline:
        raise ValueError("scalar baseline commit does not match requested baseline")
    if candidate_commit != expected_candidate:
        raise ValueError("scalar candidate commit does not match exact-batch candidate")

    rounds = report["round_evidence"]
    for index, round_entry in enumerate(rounds):
        for label, expected_commit in (
            ("baseline", baseline_commit),
            ("candidate", candidate_commit),
        ):
            environment = round_entry[label]["environment"]
            if (
                environment.get("commit") != expected_commit
                or environment.get("dirty") != "false"
            ):
                raise ValueError(
                    f"scalar round {index} {label} commit/clean evidence is invalid"
                )
    baseline_samples = [
        round_entry["baseline"]["timings_ns"]["layers"][PRIMARY_LAYER]
        for round_entry in rounds
    ]
    candidate_samples = [
        round_entry["candidate"]["timings_ns"]["layers"][PRIMARY_LAYER]
        for round_entry in rounds
    ]
    ratios = [
        right / left
        for left, right in zip(baseline_samples, candidate_samples, strict=True)
    ]
    confidence = _median_confidence(ratios, PAIRED_BOOTSTRAP_SEED)
    if confidence[1] > SCALAR_REGRESSION_LIMIT:
        raise ValueError(
            "quick scalar rangeset_public upper 95% candidate/baseline CI exceeds 1.03"
        )
    return {
        "baseline_commit": baseline_commit,
        "candidate_commit": candidate_commit,
        "rangeset_public_median_ratio": statistics.median(ratios),
        "rangeset_public_confidence_95_ratio": list(confidence),
        "threshold_upper_95": SCALAR_REGRESSION_LIMIT,
    }


def verify_artifact(
    json_path: Path,
    *,
    expected_candidate: str | None = None,
    require_samples: int | None = None,
    enforce_gates: bool = False,
    scalar_attribution: Path | None = None,
    expected_baseline: str | None = None,
    require_scalar_samples: int | None = None,
) -> dict[str, Any]:
    """Verify the canonical triplet and optionally complete promotion evidence."""
    if json_path.suffix != ".json":
        raise ValueError("exact-batch artifact must use a .json suffix")
    markdown_path = json_path.with_suffix(".md")
    checksum_path = Path(f"{json_path}.sha256")
    for path in (json_path, markdown_path, checksum_path):
        if not path.is_file():
            raise ValueError(f"exact-batch artifact is missing: {path}")
    encoded = json_path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if checksum_path.read_text(encoding="utf-8") != f"{digest}  {json_path.name}\n":
        raise ValueError("exact-batch checksum sidecar does not match JSON bytes")
    report = _strict_json(encoded)
    canonical = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    if encoded != canonical:
        raise ValueError("exact-batch JSON is not canonical")
    if markdown_path.read_text(encoding="utf-8") != _render_markdown(report, digest):
        raise ValueError("exact-batch Markdown does not match canonical JSON")
    result = verify_report(
        report,
        expected_candidate=expected_candidate,
        require_samples=require_samples,
        enforce_gates=enforce_gates,
    )
    if enforce_gates and scalar_attribution is None:
        raise ValueError("complete promotion verification requires scalar attribution")
    if scalar_attribution is not None:
        result["scalar"] = verify_scalar_attribution(
            scalar_attribution,
            expected_candidate=result["candidate_commit"],
            expected_baseline=expected_baseline,
            require_samples=require_scalar_samples,
        )
    result["json_sha256"] = digest
    result["workload_digest"] = report["workload_manifest"]["digest"]
    result["gates"] = report["gates"]
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--expected-candidate")
    parser.add_argument("--expected-baseline")
    parser.add_argument("--require-samples", type=int)
    parser.add_argument("--enforce-gates", action="store_true")
    parser.add_argument("--scalar-attribution", type=Path)
    parser.add_argument("--require-scalar-samples", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify_artifact(
        args.json_path,
        expected_candidate=args.expected_candidate,
        require_samples=args.require_samples,
        enforce_gates=args.enforce_gates,
        scalar_attribution=args.scalar_attribution,
        expected_baseline=args.expected_baseline,
        require_scalar_samples=args.require_scalar_samples,
    )
    batch16 = result["gates"]["batch16_speedup"]["observed"]
    message = (
        f"verified exact-batch candidate={result['candidate_commit']} "
        f"batch16_speedup_lower_95={batch16:.4f}"
    )
    if "scalar" in result:
        scalar_upper = result["scalar"]["rangeset_public_confidence_95_ratio"][1]
        message += f" scalar_upper_95={scalar_upper:.4f}"
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
