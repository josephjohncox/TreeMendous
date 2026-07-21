#!/usr/bin/env python3
"""Correctness-attested scaling benchmark for stable exact-batch mutations."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import sysconfig
import time
from array import array
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - evidence runs on Linux/macOS
    resource = None  # type: ignore[assignment]

from treemendous import Span, create_range_set
from treemendous.exact_batch import (
    BatchLimits,
    ExactBatchRangeSet,
    MutationOpcode,
)

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
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

Operation = tuple[int, int, int]


def _domain(interval_count: int) -> list[tuple[int, int]]:
    return [(index * 4, index * 4 + 2) for index in range(interval_count)]


def trace_for_count(interval_count: int) -> tuple[Operation, ...]:
    """Return a restorative batch targeting the last four domain components."""
    if interval_count not in INTERVAL_COUNTS:
        raise ValueError(f"unsupported interval count: {interval_count}")
    rows: list[Operation] = []
    for index in range(interval_count - 1, interval_count - 5, -1):
        start = index * 4
        end = start + 2
        rows.extend(
            (
                (MutationOpcode.DISCARD, start, end),
                (MutationOpcode.DISCARD_REQUIRE_COVERED, start, end),
                (MutationOpcode.ADD, start, end),
                (MutationOpcode.ADD, start, end),
            )
        )
    return tuple(rows)


def _packed(trace: tuple[Operation, ...]) -> bytes:
    values = array("q", (value for row in trace for value in row))
    if values.itemsize != 8:
        raise RuntimeError("native signed long long is not 64 bits")
    return values.tobytes()


def _new_exact(interval_count: int) -> ExactBatchRangeSet:
    limits = BatchLimits(**RESOURCE_LIMITS)
    if asdict(limits) != RESOURCE_LIMITS:
        raise AssertionError("scaling resource limits no longer match BatchLimits")
    return ExactBatchRangeSet(
        _domain(interval_count), initially_available=True, limits=limits
    )


def _new_scalar(interval_count: int) -> Any:
    return create_range_set(
        _domain(interval_count),
        backend=BASELINE_BACKEND,
        initially_available=True,
    )


def _scalar_results(manager: Any, trace: tuple[Operation, ...]) -> tuple[Any, ...]:
    results = []
    for opcode, start, end in trace:
        span = Span(start, end)
        if opcode == MutationOpcode.ADD:
            results.append(manager.add(span))
        else:
            results.append(
                manager.discard(
                    span,
                    require_covered=(opcode == MutationOpcode.DISCARD_REQUIRE_COVERED),
                )
            )
    return tuple(results)


def _packed_result_bytes(result: Any) -> int:
    return sum(
        view.nbytes
        for view in (
            result.changed_offsets,
            result.changed_spans,
            result.changed_lengths,
            result.fully_covered,
        )
    )


def _validate_case(
    interval_count: int, trace: tuple[Operation, ...], packed_trace: bytes
) -> tuple[Any, tuple[Any, ...], int]:
    exact = _new_exact(interval_count)
    scalar = _new_scalar(interval_count)
    exact_before = exact.snapshot()
    scalar_before = scalar.snapshot()
    if len(exact_before.intervals) != interval_count:
        raise AssertionError("exact setup has the wrong live interval count")
    if exact_before != scalar_before:
        raise AssertionError("exact and canonical initial snapshots differ")
    exact_result = exact.mutate_packed(packed_trace)
    exact_rows = exact_result.materialize()
    scalar_rows = _scalar_results(scalar, trace)
    exact_after = exact.snapshot()
    scalar_after = scalar.snapshot()
    if exact_rows != scalar_rows:
        raise AssertionError("exact and canonical per-row semantics differ")
    if exact_after != scalar_after or exact_after != exact_before:
        raise AssertionError("scaling workload does not restore the exact state")
    if len(exact_after.intervals) != interval_count:
        raise AssertionError("exact final state has the wrong live interval count")
    return exact_before, exact_rows, _packed_result_bytes(exact_result)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _median_confidence(values: list[float]) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    bootstrap = [
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return _percentile(bootstrap, 0.025), _percentile(bootstrap, 0.975)


def _checksum(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def workload_manifest() -> dict[str, Any]:
    """Return the fixed matrix and generated-operation manifest."""
    cases = {}
    for interval_count in INTERVAL_COUNTS:
        body = {
            "interval_count": interval_count,
            "domain_generator": "component_i=[4*i,4*i+2)",
            "initially_available": True,
            "operations": [list(row) for row in trace_for_count(interval_count)],
        }
        cases[str(interval_count)] = {**body, "digest": _checksum(body)}
    body = {
        "schema": WORKLOAD_SCHEMA,
        "batch_size": BATCH_SIZE,
        "interval_counts": list(INTERVAL_COUNTS),
        "cases": cases,
    }
    return {**body, "digest": _checksum(body)}


def _git_metadata() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        clean = git("status", "--porcelain") == ""
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
        clean = False
    return {"commit": commit, "clean_worktree": clean}


def _compiler_version() -> str:
    compiler = os.environ.get("CXX", "c++")
    try:
        completed = subprocess.run(
            [compiler, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.splitlines()[0] if completed.stdout else "unavailable"


def _extension_metadata(module_name: str) -> dict[str, str]:
    module = importlib.import_module(module_name)
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str):
        raise RuntimeError(f"native extension {module_name!r} has no file")
    path = Path(raw_path).resolve()
    try:
        display_path = str(path.relative_to(_REPOSITORY_ROOT))
    except ValueError:
        display_path = str(path)
    return {
        "path": display_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or 0,
    }


def _provenance() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    build_flags = {
        name: os.environ.get(name, "")
        for name in (
            "TREE_MENDOUS_DISABLE_OPTIMIZATIONS",
            "TREE_MENDOUS_LOCAL_NATIVE",
            "TREE_MENDOUS_SANITIZERS",
            "TREE_MENDOUS_GLIBCXX_DEBUG",
        )
    }
    build = {
        "command": os.environ.get(
            "TREE_MENDOUS_BUILD_COMMAND",
            "python setup.py build_ext --inplace --force",
        ),
        "cxx": _compiler_version(),
        "cc": str(sysconfig.get_config_var("CC") or "unknown"),
        "cflags": str(sysconfig.get_config_var("CFLAGS") or "unknown"),
        "build_flags": build_flags,
        "extensions": {
            "exact_batch": _extension_metadata("treemendous.cpp._exact_batch"),
            BASELINE_BACKEND: _extension_metadata("treemendous.cpp.boundary"),
        },
    }
    return _git_metadata(), _environment(), build


def _peak_rss_bytes() -> int:
    if resource is None:
        raise RuntimeError("process peak RSS is unavailable on this platform")
    usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return usage if sys.platform == "darwin" else usage * 1024


def _build_report(*, rows: dict[str, Any], samples: int) -> dict[str, Any]:
    candidate, environment, build = _provenance()
    observed = rows["100000"]["batch_latency_ns_confidence_95"][1]
    gate = {
        "threshold_ns": LATENCY_LIMIT_NS,
        "observed_upper_95_ns": observed,
        "passed": observed <= LATENCY_LIMIT_NS,
    }
    return {
        "schema": SCHEMA,
        "candidate": candidate,
        "environment": environment,
        "build": build,
        "methodology": {
            "samples": samples,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "batch_size": BATCH_SIZE,
            "interval_counts": list(INTERVAL_COUNTS),
            "timed_layer": "one mutate_packed call including staging, ordered execution, atomic commit, and packed result construction",
            "excluded": "domain/manager setup, canonical replay, result materialization, snapshots, and validation",
            "state_invariant": "every timed call begins and ends with N live intervals",
        },
        "resource_limits": dict(RESOURCE_LIMITS),
        "workload_manifest": workload_manifest(),
        "rows": rows,
        "thresholds": {"batch16_100000_upper_95_latency_ns": LATENCY_LIMIT_NS},
        "gates": {"production_envelope": gate},
        "process_peak_rss_bytes": _peak_rss_bytes(),
    }


def run_benchmark(*, samples: int = 30) -> dict[str, Any]:
    """Measure the supported scaling matrix and attest every observed result."""
    if samples < MINIMUM_SAMPLES:
        raise ValueError(
            f"scaling benchmark requires at least {MINIMUM_SAMPLES} samples"
        )

    rows: dict[str, Any] = {}
    for interval_count in INTERVAL_COUNTS:
        trace = trace_for_count(interval_count)
        packed_trace = _packed(trace)
        expected_snapshot, expected_rows, expected_result_bytes = _validate_case(
            interval_count, trace, packed_trace
        )
        manager = _new_exact(interval_count)
        latencies: list[int] = []
        for _ in range(samples):
            started = time.perf_counter_ns()
            result = manager.mutate_packed(packed_trace)
            elapsed = time.perf_counter_ns() - started
            latencies.append(elapsed)
            if result.materialize() != expected_rows:
                raise AssertionError(
                    "timed call per-row results failed canonical replay"
                )
            if _packed_result_bytes(result) != expected_result_bytes:
                raise AssertionError("timed call packed-result byte count changed")
        final_snapshot = manager.snapshot()
        if final_snapshot != expected_snapshot:
            raise AssertionError("timed calls changed the final canonical snapshot")
        if len(final_snapshot.intervals) != interval_count:
            raise AssertionError("timed calls changed the live interval count")
        latency_ci = _median_confidence([float(value) for value in latencies])
        throughput = [BATCH_SIZE * 1_000_000_000 / value for value in latencies]
        rows[str(interval_count)] = {
            "interval_count": interval_count,
            "batch_size": BATCH_SIZE,
            "batch_latency_ns_samples": latencies,
            "batch_latency_ns_median": statistics.median(latencies),
            "batch_latency_ns_confidence_95": latency_ci,
            "logical_operations_per_second_median": statistics.median(throughput),
            "logical_operations_per_second_confidence_95": _median_confidence(
                throughput
            ),
            "packed_result_bytes": expected_result_bytes,
            "process_peak_rss_bytes": _peak_rss_bytes(),
            "validated_sample_count": samples,
            "initial_and_final_interval_count": interval_count,
        }
    return _build_report(rows=rows, samples=samples)


def render_markdown(report: dict[str, Any], digest: str) -> str:
    """Render the Markdown companion bound to the canonical JSON bytes."""
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


def write_artifacts(report: dict[str, Any], output: Path) -> tuple[Path, Path, Path]:
    """Atomically write canonical JSON, Markdown, and checksum sidecar."""
    if output.suffix != ".json":
        raise ValueError("scaling output must use a .json suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    markdown = output.with_suffix(".md")
    checksum = Path(f"{output}.sha256")
    files = (
        (output, encoded),
        (markdown, render_markdown(report, digest).encode()),
        (checksum, f"{digest}  {output.name}\n".encode()),
    )
    for destination, contents in files:
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_bytes(contents)
        temporary.replace(destination)
    return output, markdown, checksum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--enforce-gate", action="store_true")
    args = parser.parse_args()
    report = run_benchmark(samples=args.samples)
    if args.output:
        write_artifacts(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(
        args.enforce_gate and not report["gates"]["production_envelope"]["passed"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
