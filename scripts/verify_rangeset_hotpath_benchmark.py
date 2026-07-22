#!/usr/bin/env python3
"""Strictly verify the range-set hot-path benchmark artifact triplet.

The verifier recomputes every stored derivation (medians, throughput 95%
confidence intervals, workload digest, canonical trace) and binds the artifact
to the loaded native binary and the recorded candidate commit.  It rejects
duplicate JSON keys, non-finite values, exact-type coercion, non-canonical
bytes, and any checksum/Markdown tamper.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

SCHEMA = "treemendous-rangeset-hotpath-benchmark-v1"
WORKLOAD_SCHEMA = "treemendous-rangeset-hotpath-workload-v1"
BACKEND = "cpp_boundary"
INITIAL_BLOCK_COUNT = 64
BLOCK_STRIDE = 32
FREE_BLOCK_LENGTH = 16
TRACE_BLOCKS = 16
DOMAIN = (0, 4_096)
THROUGHPUT_BOOTSTRAP_SEED = 16
BOOTSTRAP_RESAMPLES = 10_000
SAMPLES_MINIMUM = 20
PATHS = (
    "mutation_result_synchronized",
    "mutation_result_unsynchronized",
    "scalar_synchronized",
    "scalar_unsynchronized",
    "native_floor",
)
OP_ADD = 0
OP_DISCARD = 1
OP_DISCARD_REQUIRE_COVERED = 2
_EXTENSION_MODULE = "treemendous.cpp.boundary"
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
        raise ValueError("hot-path JSON is invalid") from exc
    if not isinstance(decoded, dict):
        raise ValueError("hot-path JSON root must be an object")
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


def _exact_json_equal(actual: Any, expected: Any) -> bool:
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


def _positive_int_samples(
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
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _median_confidence(values: list[float]) -> tuple[float, float]:
    rng = random.Random(THROUGHPUT_BOOTSTRAP_SEED)
    boot = [
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return _percentile(boot, 0.025), _percentile(boot, 0.975)


def _initial_intervals() -> list[list[int]]:
    return [
        [index * BLOCK_STRIDE, index * BLOCK_STRIDE + FREE_BLOCK_LENGTH]
        for index in range(INITIAL_BLOCK_COUNT)
    ]


def _trace() -> list[list[int]]:
    operations: list[list[int]] = []
    for index in range(TRACE_BLOCKS):
        base = index * BLOCK_STRIDE
        operations.append([OP_DISCARD, base + 4, base + 8])
        operations.append([OP_ADD, base + 4, base + 8])
        operations.append([OP_DISCARD_REQUIRE_COVERED, base + 12, base + 20])
        operations.append([OP_ADD, base, base + FREE_BLOCK_LENGTH])
    return operations


def _expected_changed_lengths() -> list[int]:
    """Independently recompute the canonical per-operation changed lengths."""
    free = set(
        point for start, end in _initial_intervals() for point in range(start, end)
    )
    lengths: list[int] = []
    for opcode, start, end in _trace():
        target = set(range(start, end))
        if opcode == OP_ADD:
            changed = target - free
            free |= target
            lengths.append(len(changed))
        else:
            covered = target <= free
            if opcode == OP_DISCARD_REQUIRE_COVERED and not covered:
                lengths.append(0)
                continue
            changed = target & free
            free -= target
            lengths.append(len(changed))
    return lengths


def _expected_manifest() -> dict[str, Any]:
    body = {
        "schema": WORKLOAD_SCHEMA,
        "domain": list(DOMAIN),
        "initial_intervals": _initial_intervals(),
        "operations": _trace(),
        "expected_changed_lengths": _expected_changed_lengths(),
    }
    return {**body, "digest": _checksum(body)}


def _loaded_extension_metadata() -> dict[str, str]:
    module = importlib.import_module(_EXTENSION_MODULE)
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str):
        raise ValueError(f"loaded native extension {_EXTENSION_MODULE!r} has no file")
    try:
        path = Path(raw_path).resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"loaded native extension {_EXTENSION_MODULE!r} file is unavailable"
        ) from exc
    try:
        display_path = str(path.relative_to(_REPOSITORY_ROOT))
    except ValueError:
        display_path = str(path)
    return {
        "path": display_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _render_markdown(report: dict[str, Any], digest: str) -> str:
    lines = [
        "# Range-set hot-path benchmark",
        "",
        f"- Candidate: `{report['candidate']['commit']}`",
        f"- Clean worktree: `{str(report['candidate']['clean_worktree']).lower()}`",
        f"- Backend: `{report['backend']}`",
        f"- Host: `{report['environment']['platform']}`",
        f"- Workload digest: `{report['workload_manifest']['digest']}`",
        f"- JSON SHA-256: `{digest}`",
        "",
        "Not a universal claim: one interface family, one restorative workload, "
        "one host, one timed layer.",
        "",
        "| Path | Median ns/op | Median M ops/s | 95% CI M ops/s |",
        "|---|---:|---:|---:|",
    ]
    for name in PATHS:
        row = report["paths"][name]
        low, high = row["ops_per_second_confidence_95"]
        lines.append(
            f"| {name} | {row['median_ns_per_operation']} | "
            f"{row['median_ops_per_second'] / 1e6:.3f} | "
            f"{low / 1e6:.3f}\u2013{high / 1e6:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _verify_metadata(report: dict[str, Any], expected_candidate: str | None) -> str:
    candidate = _mapping(report.get("candidate"), "candidate")
    _exact_fields(candidate, {"commit", "clean_worktree"}, "candidate")
    commit = candidate.get("commit")
    if not _valid_commit(commit):
        raise ValueError("candidate commit is invalid")
    if expected_candidate is not None and commit != expected_candidate:
        raise ValueError("candidate commit does not match")
    if not _boolean(candidate.get("clean_worktree"), "candidate cleanliness"):
        raise ValueError("hot-path artifact was produced from a dirty worktree")

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
    _exact_fields(
        flags,
        {
            "TREE_MENDOUS_DISABLE_OPTIMIZATIONS",
            "TREE_MENDOUS_LOCAL_NATIVE",
            "TREE_MENDOUS_SANITIZERS",
            "TREE_MENDOUS_GLIBCXX_DEBUG",
        },
        "build flags",
    )
    if any(not isinstance(value, str) for value in flags.values()):
        raise ValueError("build flags are invalid")
    extensions = _mapping(build["extensions"], "extensions")
    _exact_fields(extensions, {BACKEND}, "extensions")
    extension = _mapping(extensions[BACKEND], f"{BACKEND} extension")
    _exact_fields(extension, {"path", "sha256"}, f"{BACKEND} extension")
    if not isinstance(extension["path"], str) or not extension["path"]:
        raise ValueError(f"{BACKEND} extension path is invalid")
    if not _valid_sha256(extension["sha256"]):
        raise ValueError(f"{BACKEND} extension digest is invalid")
    loaded = _loaded_extension_metadata()
    if extension["path"] != loaded["path"]:
        raise ValueError(f"{BACKEND} extension path does not match loaded module")
    if extension["sha256"] != loaded["sha256"]:
        raise ValueError(f"{BACKEND} extension digest does not match loaded module")
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
            "workload",
            "timed_layer",
            "excluded",
            "throughput_bootstrap_seed",
            "bootstrap_resamples",
            "paths",
            "universal_claim",
        },
        "methodology",
    )
    samples = methodology.get("samples")
    if (
        isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples < SAMPLES_MINIMUM
    ):
        raise ValueError("benchmark sample count is invalid")
    if require_samples is not None and samples != require_samples:
        raise ValueError("benchmark sample count does not match the requirement")
    fixed = {
        "throughput_bootstrap_seed": THROUGHPUT_BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "paths": list(PATHS),
        "universal_claim": False,
    }
    if any(
        not _exact_json_equal(methodology.get(key), value)
        for key, value in fixed.items()
    ):
        raise ValueError("benchmark methodology is inconsistent")
    for field in ("workload", "timed_layer", "excluded"):
        if not isinstance(methodology[field], str) or not methodology[field]:
            raise ValueError("benchmark methodology narrative is invalid")
    target = methodology.get("target_operations")
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError("target operation count is invalid")
    return samples, target


def _verify_paths(report: dict[str, Any], samples: int, target_operations: int) -> None:
    paths = _mapping(report.get("paths"), "paths")
    if set(paths) != set(PATHS):
        raise ValueError("benchmark paths are incomplete")
    operations_per_trace = len(_trace())
    expected_logical = max(SAMPLES_MINIMUM, target_operations // operations_per_trace)
    for name in PATHS:
        description = f"path {name!r}"
        row = _mapping(paths[name], description)
        _exact_fields(
            row,
            {
                "interface",
                "operations_per_trace",
                "logical_operations_per_sample",
                "ns_per_operation_samples",
                "median_ns_per_operation",
                "median_ops_per_second",
                "ops_per_second_confidence_95",
            },
            description,
        )
        if not isinstance(row["interface"], str) or not row["interface"]:
            raise ValueError(f"{description} interface label is invalid")
        declaration = {
            "operations_per_trace": operations_per_trace,
            "logical_operations_per_sample": expected_logical * operations_per_trace,
        }
        if any(
            not _exact_json_equal(row.get(key), value)
            for key, value in declaration.items()
        ):
            raise ValueError(f"{description} declaration is invalid")
        ns_samples = _positive_int_samples(
            row.get("ns_per_operation_samples"),
            f"{description} ns/op samples",
            expected_count=samples,
        )
        if not _exact_json_equal(
            row.get("median_ns_per_operation"), statistics.median(ns_samples)
        ):
            raise ValueError(f"{description} median ns/op is inconsistent")
        throughput = [1_000_000_000 / value for value in ns_samples]
        if not _exact_json_equal(
            row.get("median_ops_per_second"), statistics.median(throughput)
        ):
            raise ValueError(f"{description} median throughput is inconsistent")
        if not _exact_json_equal(
            row.get("ops_per_second_confidence_95"),
            list(_median_confidence(throughput)),
        ):
            raise ValueError(f"{description} throughput CI is inconsistent")


def verify_report(
    report: dict[str, Any],
    *,
    expected_candidate: str | None = None,
    require_samples: int | None = None,
) -> dict[str, Any]:
    _exact_fields(
        report,
        {
            "schema",
            "candidate",
            "environment",
            "build",
            "backend",
            "methodology",
            "workload_manifest",
            "paths",
        },
        "report",
    )
    if report.get("schema") != SCHEMA:
        raise ValueError("unexpected hot-path schema")
    if report.get("backend") != BACKEND:
        raise ValueError("hot-path backend must be cpp_boundary")
    commit = _verify_metadata(report, expected_candidate)
    samples, target_operations = _verify_methodology(report, require_samples)
    if not _exact_json_equal(report.get("workload_manifest"), _expected_manifest()):
        raise ValueError("restorative workload manifest or digest is inconsistent")
    _verify_paths(report, samples, target_operations)
    return {"candidate_commit": commit, "samples": samples}


def verify_artifact(
    json_path: Path,
    *,
    expected_candidate: str | None = None,
    require_samples: int | None = None,
) -> dict[str, Any]:
    if json_path.suffix != ".json":
        raise ValueError("hot-path artifact must use a .json suffix")
    markdown_path = json_path.with_suffix(".md")
    checksum_path = Path(f"{json_path}.sha256")
    for path in (json_path, markdown_path, checksum_path):
        if not path.is_file():
            raise ValueError(f"hot-path artifact is missing: {path}")
    encoded = json_path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if checksum_path.read_text(encoding="utf-8") != f"{digest}  {json_path.name}\n":
        raise ValueError("hot-path checksum sidecar does not match JSON bytes")
    report = _strict_json(encoded)
    canonical = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    if encoded != canonical:
        raise ValueError("hot-path JSON is not canonical")
    if markdown_path.read_text(encoding="utf-8") != _render_markdown(report, digest):
        raise ValueError("hot-path Markdown does not match canonical JSON")
    result = verify_report(
        report,
        expected_candidate=expected_candidate,
        require_samples=require_samples,
    )
    result["json_sha256"] = digest
    result["workload_digest"] = report["workload_manifest"]["digest"]
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--expected-candidate")
    parser.add_argument("--require-samples", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify_artifact(
        args.json_path,
        expected_candidate=args.expected_candidate,
        require_samples=args.require_samples,
    )
    print(
        f"verified rangeset hot-path candidate={result['candidate_commit']} "
        f"samples={result['samples']} json_sha256={result['json_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
