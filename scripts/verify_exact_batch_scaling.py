#!/usr/bin/env python3
"""Strictly verify stable exact-batch scaling evidence and its fixed gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
from pathlib import Path
from typing import Any

SCHEMA = "treemendous-exact-batch-scaling-v1"
WORKLOAD_SCHEMA = "treemendous-exact-batch-scaling-workload-v1"
INTERVAL_COUNTS = (64, 1_000, 10_000, 100_000)
BATCH_SIZE = 16
MINIMUM_SAMPLES = 20
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 100_016
LATENCY_LIMIT_NS = 10_000_000
BASELINE_BACKEND = "cpp_boundary"
RESOURCE_LIMITS = {
    "max_operations": 1_000_000,
    "max_live_intervals": 100_000,
    "max_changed_spans": 2_000_000,
    "max_result_bytes": 256 * 1024 * 1024,
    "max_work_units": 100_000_000,
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
        raise ValueError("scaling JSON is invalid") from exc
    if not isinstance(decoded, dict):
        raise ValueError("scaling JSON root must be an object")
    return decoded


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    return value


def _exact_fields(value: dict[str, Any], expected: set[str], description: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{description} fields are incomplete or unexpected")


def _exact_json_equal(actual: Any, expected: Any) -> bool:
    """Compare decoded JSON without Python's bool/int or int/float coercions."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _exact_json_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_json_equal(item, value)
            for item, value in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


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


def _median_confidence(values: list[float]) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    bootstrap = [
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return _percentile(bootstrap, 0.025), _percentile(bootstrap, 0.975)


def _trace(interval_count: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for index in range(interval_count - 1, interval_count - 5, -1):
        start = index * 4
        end = start + 2
        rows.extend(
            ([1, start, end], [2, start, end], [0, start, end], [0, start, end])
        )
    return rows


def _expected_manifest() -> dict[str, Any]:
    cases = {}
    for interval_count in INTERVAL_COUNTS:
        body = {
            "interval_count": interval_count,
            "domain_generator": "component_i=[4*i,4*i+2)",
            "initially_available": True,
            "operations": _trace(interval_count),
        }
        cases[str(interval_count)] = {**body, "digest": _checksum(body)}
    body = {
        "schema": WORKLOAD_SCHEMA,
        "batch_size": BATCH_SIZE,
        "interval_counts": list(INTERVAL_COUNTS),
        "cases": cases,
    }
    return {**body, "digest": _checksum(body)}


def _current_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or 0,
    }


def _repository_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("repository provenance is unavailable") from exc
    return {"commit": commit, "clean_worktree": status == ""}


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


def _verify_provenance(report: dict[str, Any], expected_candidate: str | None) -> str:
    candidate = _mapping(report.get("candidate"), "candidate")
    _exact_fields(candidate, {"commit", "clean_worktree"}, "candidate")
    commit = candidate.get("commit")
    if not _valid_commit(commit):
        raise ValueError("candidate commit is invalid")
    clean_worktree = candidate.get("clean_worktree")
    if type(clean_worktree) is not bool or not clean_worktree:
        raise ValueError("scaling artifact was produced from a dirty worktree")
    if expected_candidate is not None and commit != expected_candidate:
        raise ValueError("candidate commit does not match the required SHA")
    repository = _repository_state()
    repository_clean = repository.get("clean_worktree")
    if type(repository_clean) is not bool or not repository_clean:
        raise ValueError("scaling verification requires a clean worktree")
    if repository.get("commit") != commit:
        raise ValueError("candidate commit does not match the verification checkout")

    environment = _mapping(report.get("environment"), "environment")
    _exact_fields(environment, set(_current_environment()), "environment")
    if not _exact_json_equal(environment, _current_environment()):
        raise ValueError("recorded environment does not match verification runtime")

    build = _mapping(report.get("build"), "build")
    _exact_fields(
        build,
        {"command", "cxx", "cc", "cflags", "build_flags", "extensions"},
        "build",
    )
    for field in ("command", "cxx", "cc", "cflags"):
        if not isinstance(build.get(field), str) or not build[field]:
            raise ValueError("compiler/build provenance is invalid")
    flags = _mapping(build.get("build_flags"), "build flags")
    expected_flags = {
        "TREE_MENDOUS_DISABLE_OPTIMIZATIONS",
        "TREE_MENDOUS_LOCAL_NATIVE",
        "TREE_MENDOUS_SANITIZERS",
        "TREE_MENDOUS_GLIBCXX_DEBUG",
    }
    _exact_fields(flags, expected_flags, "build flags")
    if any(not isinstance(value, str) for value in flags.values()):
        raise ValueError("build flags are invalid")
    extensions = _mapping(build.get("extensions"), "extensions")
    _exact_fields(extensions, set(_EXTENSION_MODULES), "extensions")
    for name, module_name in _EXTENSION_MODULES.items():
        extension = _mapping(extensions.get(name), f"{name} extension")
        _exact_fields(extension, {"path", "sha256"}, f"{name} extension")
        if not isinstance(extension.get("path"), str) or not extension["path"]:
            raise ValueError(f"{name} extension path is invalid")
        if not _valid_sha256(extension.get("sha256")):
            raise ValueError(f"{name} extension digest is invalid")
        loaded = _loaded_extension_metadata(module_name)
        if not _exact_json_equal(extension, loaded):
            raise ValueError(f"{name} extension does not match the loaded binary")
    return str(commit)


def _positive_samples(value: Any, description: str, count: int) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item <= 0
            or item > 2**63 - 1
            for item in value
        )
    ):
        raise ValueError(
            f"{description} must contain exactly {count} positive int64 samples"
        )
    return value


def _verify_methodology(report: dict[str, Any], require_samples: int | None) -> int:
    methodology = _mapping(report.get("methodology"), "methodology")
    expected_fixed = {
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "batch_size": BATCH_SIZE,
        "interval_counts": list(INTERVAL_COUNTS),
        "timed_layer": "one mutate_packed call including staging, ordered execution, atomic commit, and packed result construction",
        "excluded": "domain/manager setup, canonical replay, result materialization, snapshots, and validation",
        "state_invariant": "every timed call begins and ends with N live intervals",
    }
    _exact_fields(methodology, {"samples", *expected_fixed}, "methodology")
    samples = methodology.get("samples")
    if (
        isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples < MINIMUM_SAMPLES
    ):
        raise ValueError("scaling sample count is invalid")
    if require_samples is not None and samples != require_samples:
        raise ValueError("scaling sample count does not match the requirement")
    if any(
        not _exact_json_equal(methodology.get(key), value)
        for key, value in expected_fixed.items()
    ):
        raise ValueError("scaling methodology or matrix is inconsistent")
    return samples


def _verify_rows(report: dict[str, Any], samples: int) -> dict[str, dict[str, Any]]:
    rows = _mapping(report.get("rows"), "rows")
    if set(rows) != {str(value) for value in INTERVAL_COUNTS}:
        raise ValueError("scaling matrix rows are incomplete or unexpected")
    expected_fields = {
        "interval_count",
        "batch_size",
        "batch_latency_ns_samples",
        "batch_latency_ns_median",
        "batch_latency_ns_confidence_95",
        "logical_operations_per_second_median",
        "logical_operations_per_second_confidence_95",
        "packed_result_bytes",
        "process_peak_rss_bytes",
        "validated_sample_count",
        "initial_and_final_interval_count",
    }
    verified: dict[str, dict[str, Any]] = {}
    previous_rss = 0
    for interval_count in INTERVAL_COUNTS:
        description = f"{interval_count}-interval row"
        row = _mapping(rows.get(str(interval_count)), description)
        _exact_fields(row, expected_fields, description)
        declaration = {
            "interval_count": interval_count,
            "batch_size": BATCH_SIZE,
            "validated_sample_count": samples,
            "initial_and_final_interval_count": interval_count,
            "packed_result_bytes": 408,
        }
        if any(
            not _exact_json_equal(row.get(key), value)
            for key, value in declaration.items()
        ):
            raise ValueError(f"{description} declaration is inconsistent")
        raw = _positive_samples(
            row.get("batch_latency_ns_samples"),
            f"{description} raw latency",
            samples,
        )
        latency_values = [value / 1 for value in raw]
        latency_ci = list(_median_confidence(latency_values))
        throughput = [BATCH_SIZE * 1_000_000_000 / value for value in raw]
        if not _exact_json_equal(
            row.get("batch_latency_ns_median"), statistics.median(raw)
        ):
            raise ValueError(f"{description} latency median is inconsistent")
        if not _exact_json_equal(row.get("batch_latency_ns_confidence_95"), latency_ci):
            raise ValueError(f"{description} latency CI is inconsistent")
        if not _exact_json_equal(
            row.get("logical_operations_per_second_median"),
            statistics.median(throughput),
        ):
            raise ValueError(f"{description} throughput median is inconsistent")
        if not _exact_json_equal(
            row.get("logical_operations_per_second_confidence_95"),
            list(_median_confidence(throughput)),
        ):
            raise ValueError(f"{description} throughput CI is inconsistent")
        rss = row.get("process_peak_rss_bytes")
        if (
            isinstance(rss, bool)
            or not isinstance(rss, int)
            or rss <= 0
            or rss < previous_rss
        ):
            raise ValueError(f"{description} process peak RSS is invalid")
        previous_rss = rss
        verified[str(interval_count)] = row
    process_rss = report.get("process_peak_rss_bytes")
    if (
        isinstance(process_rss, bool)
        or not isinstance(process_rss, int)
        or process_rss < previous_rss
    ):
        raise ValueError("report process peak RSS is invalid")
    return verified


def _verify_gate(report: dict[str, Any], rows: dict[str, dict[str, Any]]) -> None:
    expected_thresholds = {"batch16_100000_upper_95_latency_ns": LATENCY_LIMIT_NS}
    if not _exact_json_equal(report.get("thresholds"), expected_thresholds):
        raise ValueError("fixed scaling threshold is missing or inconsistent")
    observed = rows["100000"]["batch_latency_ns_confidence_95"][1]
    expected_gate = {
        "threshold_ns": LATENCY_LIMIT_NS,
        "observed_upper_95_ns": observed,
        "passed": observed <= LATENCY_LIMIT_NS,
    }
    gates = _mapping(report.get("gates"), "gates")
    _exact_fields(gates, {"production_envelope"}, "gates")
    if not _exact_json_equal(gates.get("production_envelope"), expected_gate):
        raise ValueError("production envelope gate derivation is inconsistent")


def _render_markdown(report: dict[str, Any], digest: str) -> str:
    gate = report["gates"]["production_envelope"]
    lines = [
        "# Stable exact-batch scaling evidence",
        "",
        f"- Candidate: `{report['candidate']['commit']}`",
        f"- Clean worktree: `{str(report['candidate']['clean_worktree']).lower()}`",
        f"- Workload digest: `{report['workload_manifest']['digest']}`",
        f"- JSON SHA-256: `{digest}`",
        "",
        "| Live intervals | Batch | Median latency (ms) | 95% median CI (ms) | Logical ops/s | Peak RSS (MiB) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for interval_count in INTERVAL_COUNTS:
        row = report["rows"].get(str(interval_count))
        if row is None:
            continue
        confidence = row["batch_latency_ns_confidence_95"]
        lines.append(
            f"| {interval_count} | {BATCH_SIZE} | "
            f"{row['batch_latency_ns_median'] / 1_000_000:.3f} | "
            f"[{confidence[0] / 1_000_000:.3f}, {confidence[1] / 1_000_000:.3f}] | "
            f"{row['logical_operations_per_second_median']:.0f} | "
            f"{row['process_peak_rss_bytes'] / (1024 * 1024):.1f} |"
        )
    lines.extend(
        [
            "",
            f"Production envelope gate: upper 95% median latency "
            f"{gate['observed_upper_95_ns'] / 1_000_000:.3f} ms <= "
            f"{gate['threshold_ns'] / 1_000_000:.3f} ms "
            f"({'pass' if gate['passed'] else 'fail'}).",
        ]
    )
    return "\n".join(lines) + "\n"


def verify_report(
    report: dict[str, Any],
    *,
    expected_candidate: str | None = None,
    require_samples: int | None = None,
    enforce_gate: bool = False,
) -> dict[str, Any]:
    """Verify all fields, fixed inputs, raw samples, and derived values."""
    _exact_fields(
        report,
        {
            "schema",
            "candidate",
            "environment",
            "build",
            "methodology",
            "resource_limits",
            "workload_manifest",
            "rows",
            "thresholds",
            "gates",
            "process_peak_rss_bytes",
        },
        "report",
    )
    if report.get("schema") != SCHEMA:
        raise ValueError("unexpected scaling schema")
    commit = _verify_provenance(report, expected_candidate)
    samples = _verify_methodology(report, require_samples)
    if not _exact_json_equal(report.get("resource_limits"), RESOURCE_LIMITS):
        raise ValueError("scaling resource limits are inconsistent")
    if not _exact_json_equal(report.get("workload_manifest"), _expected_manifest()):
        raise ValueError("scaling workload manifest or digest is inconsistent")
    rows = _verify_rows(report, samples)
    _verify_gate(report, rows)
    gate = report["gates"]["production_envelope"]
    if enforce_gate and not gate["passed"]:
        raise ValueError("fixed production envelope latency gate failed")
    return {
        "candidate_commit": commit,
        "samples": samples,
        "workload_digest": report["workload_manifest"]["digest"],
        "gate": gate,
    }


def verify_artifact(
    json_path: Path,
    *,
    expected_candidate: str | None = None,
    require_samples: int | None = None,
    enforce_gate: bool = False,
) -> dict[str, Any]:
    """Verify the canonical scaling JSON/Markdown/checksum triplet."""
    if json_path.suffix != ".json":
        raise ValueError("scaling artifact must use a .json suffix")
    markdown_path = json_path.with_suffix(".md")
    checksum_path = Path(f"{json_path}.sha256")
    for path in (json_path, markdown_path, checksum_path):
        if not path.is_file():
            raise ValueError(f"scaling artifact is missing: {path}")
    encoded = json_path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if checksum_path.read_text(encoding="utf-8") != f"{digest}  {json_path.name}\n":
        raise ValueError("scaling checksum sidecar does not match JSON bytes")
    report = _strict_json(encoded)
    canonical = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    if encoded != canonical:
        raise ValueError("scaling JSON is not canonical")
    if markdown_path.read_text(encoding="utf-8") != _render_markdown(report, digest):
        raise ValueError("scaling Markdown does not match canonical JSON")
    result = verify_report(
        report,
        expected_candidate=expected_candidate,
        require_samples=require_samples,
        enforce_gate=enforce_gate,
    )
    result["json_sha256"] = digest
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--expected-candidate")
    parser.add_argument("--require-samples", type=int)
    parser.add_argument("--enforce-gate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify_artifact(
        args.json_path,
        expected_candidate=args.expected_candidate,
        require_samples=args.require_samples,
        enforce_gate=args.enforce_gate,
    )
    gate = result["gate"]
    print(
        f"verified exact-batch scaling candidate={result['candidate_commit']} "
        f"batch16_100000_upper_95_ms="
        f"{gate['observed_upper_95_ns'] / 1_000_000:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
