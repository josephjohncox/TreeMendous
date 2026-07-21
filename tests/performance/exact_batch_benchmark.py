#!/usr/bin/env python3
"""Paired benchmark for the stable exact whole-batch CPU mutation path."""

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
from pathlib import Path
from typing import Any

from tests.performance.mutation_attribution import paired_statistics
from treemendous import BackendUnavailableError, Span, create_range_set
from treemendous.exact_batch import ExactBatchRangeSet, MutationOpcode

BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
DOMAIN = (0, 1_024)
BASELINE_BACKEND = "cpp_boundary"
INITIAL_INTERVAL_COUNT = 64
ABSOLUTE_BATCH16_OPS_PER_SECOND = 2_000_000.0
BATCH16_SPEEDUP = 2.0
SCALAR_REGRESSION_LIMIT = 1.03
SCHEMA = "treemendous-exact-batch-benchmark-v3"
WORKLOAD_SCHEMA = "treemendous-exact-batch-restorative-workload-v1"
BOOTSTRAP_RESAMPLES = 10_000
PAIRED_BOOTSTRAP_SEED = 50
THROUGHPUT_BOOTSTRAP_SEED = 16
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

Operation = tuple[MutationOpcode, Span]
Trace = tuple[Operation, ...]


def _multi_component_block(start_index: int) -> Trace:
    """Fill three gaps across four components, then restore those exact gaps."""
    base = start_index * 16
    return (
        (MutationOpcode.ADD, Span(base, base + 56)),
        (MutationOpcode.DISCARD, Span(base + 8, base + 16)),
        (MutationOpcode.DISCARD, Span(base + 24, base + 32)),
        (MutationOpcode.DISCARD, Span(base + 40, base + 48)),
    )


def _restorative_trace64() -> Trace:
    """Return sixteen independently restorative four-operation blocks."""
    blocks: list[Trace] = [
        (
            (MutationOpcode.DISCARD, Span(2, 6)),  # real mutation
            (MutationOpcode.ADD, Span(2, 6)),  # exact restoration
            (MutationOpcode.DISCARD_REQUIRE_COVERED, Span(8, 12)),  # rejection
            (MutationOpcode.DISCARD, Span(8, 12)),  # no-op
        ),
        _multi_component_block(0),
        (
            (MutationOpcode.DISCARD, Span(132, 140)),  # partial overlap
            (MutationOpcode.ADD, Span(132, 136)),  # exact restoration
            (MutationOpcode.ADD, Span(128, 136)),  # no-op
            (MutationOpcode.ADD, Span(128, 136)),  # duplicate no-op
        ),
        (
            (MutationOpcode.DISCARD_REQUIRE_COVERED, Span(264, 268)),
            (MutationOpcode.DISCARD_REQUIRE_COVERED, Span(264, 268)),
            (MutationOpcode.DISCARD, Span(258, 262)),
            (MutationOpcode.ADD, Span(258, 262)),
        ),
    ]
    for block_index in range(4, 16):
        if block_index % 2:
            start_index = ((block_index * 7) % 13) * 4
            blocks.append(_multi_component_block(start_index))
            continue
        interval_index = (block_index * 11) % INITIAL_INTERVAL_COUNT
        base = interval_index * 16
        blocks.append(
            (
                (MutationOpcode.DISCARD, Span(base + 1, base + 7)),
                (MutationOpcode.ADD, Span(base + 1, base + 7)),
                (MutationOpcode.ADD, Span(base, base + 8)),
                (
                    MutationOpcode.DISCARD_REQUIRE_COVERED,
                    Span(base + 8, base + 13),
                ),
            )
        )
    return tuple(operation for block in blocks for operation in block)


_TRACE64 = _restorative_trace64()


def trace_for_size(batch_size: int) -> Trace:
    """Return a deterministic trace that preserves the 64-interval state."""
    if batch_size not in BATCH_SIZES:
        raise ValueError(f"unsupported batch size: {batch_size}")
    if batch_size == 1:
        # A restorative one-operation workload can only be a no-op. Keep it as a
        # separately labelled call-overhead diagnostic, not promotion evidence.
        return ((MutationOpcode.ADD, Span(0, 8)),)
    return _TRACE64[:batch_size]


def mixed_trace() -> Trace:
    """Return the restorative 64-operation mixed trace (compatibility helper)."""
    return trace_for_size(64)


def _packed(trace: Trace) -> bytes:
    values = array(
        "q",
        (value for opcode, span in trace for value in (opcode, span.start, span.end)),
    )
    if values.itemsize != 8:
        raise RuntimeError("native signed long long is not 64 bits")
    return values.tobytes()


def _new_exact() -> ExactBatchRangeSet:
    manager = ExactBatchRangeSet(DOMAIN, initially_available=False)
    setup = tuple(
        (MutationOpcode.ADD, Span(index * 16, index * 16 + 8))
        for index in range(INITIAL_INTERVAL_COUNT)
    )
    manager.mutate_packed(_packed(setup))
    return manager


def _new_scalar() -> Any:
    manager = create_range_set(
        DOMAIN, backend=BASELINE_BACKEND, initially_available=False
    )
    for index in range(INITIAL_INTERVAL_COUNT):
        manager.add(Span(index * 16, index * 16 + 8))
    return manager


def _scalar_once(manager: Any, trace: Trace) -> None:
    for opcode, span in trace:
        if opcode == MutationOpcode.ADD:
            result = manager.add(span)
        else:
            result = manager.discard(
                span,
                require_covered=opcode == MutationOpcode.DISCARD_REQUIRE_COVERED,
            )
        del result


def _scalar_results(manager: Any, trace: Trace) -> tuple[Any, ...]:
    results = []
    for opcode, span in trace:
        if opcode == MutationOpcode.ADD:
            results.append(manager.add(span))
        else:
            results.append(
                manager.discard(
                    span,
                    require_covered=opcode == MutationOpcode.DISCARD_REQUIRE_COVERED,
                )
            )
    return tuple(results)


def _assert_initial_state(snapshot: Any, *, label: str) -> None:
    if len(snapshot.intervals) != INITIAL_INTERVAL_COUNT:
        raise AssertionError(
            f"{label} contracted to {len(snapshot.intervals)} intervals; "
            f"expected {INITIAL_INTERVAL_COUNT}"
        )


def _validate(trace: Trace, packed_trace: bytes) -> Any:
    """Validate exact per-row semantics and the restorative state invariant."""
    exact = _new_exact()
    scalar = _new_scalar()
    exact_before = exact.snapshot()
    scalar_before = scalar.snapshot()
    _assert_initial_state(exact_before, label="exact pre-trace state")
    _assert_initial_state(scalar_before, label=f"{BASELINE_BACKEND} pre-trace state")
    if exact_before != scalar_before:
        raise AssertionError("exact and cpp_boundary initial states differ")

    exact_results = exact.mutate_packed(packed_trace).materialize()
    scalar_results = _scalar_results(scalar, trace)
    exact_after = exact.snapshot()
    scalar_after = scalar.snapshot()
    if exact_results != scalar_results:
        raise AssertionError("packed and cpp_boundary per-row results differ")
    if exact_after != scalar_after:
        raise AssertionError("packed and cpp_boundary final states differ")
    if exact_after != exact_before:
        raise AssertionError("benchmark trace is not exactly restorative")
    _assert_initial_state(exact_after, label="exact post-trace state")
    _assert_initial_state(scalar_after, label=f"{BASELINE_BACKEND} post-trace state")
    return exact_before


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    try:
        lower = int(position)
    except (OverflowError, ValueError) as exc:
        raise ValueError("percentile position must be finite") from exc
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _median_confidence(
    values: list[float], seed: int = THROUGHPUT_BOOTSTRAP_SEED
) -> tuple[float, float]:
    rng = random.Random(seed)
    boot = [
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return _percentile(boot, 0.025), _percentile(boot, 0.975)


def _checksum(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def restorative_workload_manifest() -> dict[str, Any]:
    """Return the versioned, exact workload manifest and its canonical digests."""
    traces: dict[str, Any] = {}
    for batch_size in BATCH_SIZES:
        trace = {
            "classification": (
                "single-call-no-op-diagnostic" if batch_size == 1 else "restorative"
            ),
            "operations": [
                [opcode.value, span.start, span.end]
                for opcode, span in trace_for_size(batch_size)
            ],
        }
        traces[str(batch_size)] = {**trace, "digest": _checksum(trace)}
    body = {
        "schema": WORKLOAD_SCHEMA,
        "domain": list(DOMAIN),
        "initial_intervals": [
            [index * 16, index * 16 + 8] for index in range(INITIAL_INTERVAL_COUNT)
        ],
        "traces": traces,
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
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    try:
        clean = git("status", "--porcelain") == ""
    except (OSError, subprocess.CalledProcessError):
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


def _provenance() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidate = _git_metadata()
    environment = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or 0,
    }
    flags = {
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
        "build_flags": flags,
        "extensions": {
            "exact_batch": _extension_metadata("treemendous.cpp._exact_batch"),
            BASELINE_BACKEND: _extension_metadata("treemendous.cpp.boundary"),
        },
    }
    return candidate, environment, build


def _build_report(
    *,
    rows: dict[str, Any],
    materialize_samples: list[int],
    samples: int,
    target_operations: int,
) -> dict[str, Any]:
    batch16 = rows["16"]
    thresholds = {
        "batch16_ops_per_second_lower_95": ABSOLUTE_BATCH16_OPS_PER_SECOND,
        "batch16_speedup_lower_95": BATCH16_SPEEDUP,
        "batch4_speedup_lower_95": 1.0,
        "stable_scalar_candidate_over_baseline_upper_95": SCALAR_REGRESSION_LIMIT,
    }
    gates = {
        "batch16_absolute": {
            "threshold": thresholds["batch16_ops_per_second_lower_95"],
            "observed": batch16["batch_ops_per_second_confidence_95"][0],
            "passed": batch16["batch_ops_per_second_confidence_95"][0]
            >= thresholds["batch16_ops_per_second_lower_95"],
        },
        "batch16_speedup": {
            "threshold": thresholds["batch16_speedup_lower_95"],
            "observed": batch16["speedup_confidence_95"][0],
            "passed": batch16["speedup_confidence_95"][0]
            >= thresholds["batch16_speedup_lower_95"],
        },
        "break_even_by_4": {
            "threshold": thresholds["batch4_speedup_lower_95"],
            "observed": rows["4"]["speedup_confidence_95"][0],
            "passed": rows["4"]["speedup_confidence_95"][0]
            >= thresholds["batch4_speedup_lower_95"],
        },
    }
    candidate, environment, build = _provenance()
    return {
        "schema": SCHEMA,
        "candidate": candidate,
        "environment": environment,
        "build": build,
        "baseline_backend": BASELINE_BACKEND,
        "methodology": {
            "samples": samples,
            "target_operations": target_operations,
            "paired_order": "alternating within each paired sample",
            "paired_bootstrap_seed": PAIRED_BOOTSTRAP_SEED,
            "throughput_bootstrap_seed": THROUGHPUT_BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "initial_and_final_interval_count": INITIAL_INTERVAL_COUNT,
            "batch_sizes": list(BATCH_SIZES),
            "workload": "deterministic per-size restorative traces; four-operation blocks preserve the exact initial state",
            "batch_1": "separately labelled no-op call-overhead diagnostic",
            "timed_batch_layer": "buffer acquisition/copy, state staging, ordered execution, packed allocation, atomic commit, and packed-result destruction except one retained validation result per sample",
            "post_timing_validation": "the last timed packed result and both timed instances are checked against canonical per-row results and final state outside timing",
            "excluded": "manager/setup construction, invariant snapshots, post-timing validation, and materialize",
        },
        "workload_manifest": restorative_workload_manifest(),
        "rows": rows,
        "materialize_16_ns_per_operation_samples": materialize_samples,
        "thresholds": thresholds,
        "gates": gates,
    }


def run_benchmark(
    *, samples: int = 30, target_operations: int = 20_000
) -> dict[str, Any]:
    """Run paired timings against canonical public ``cpp_boundary`` calls."""
    if samples < 20:
        raise ValueError("paired benchmark requires at least 20 samples")
    if target_operations < 1:
        raise ValueError("target_operations must be positive")

    # Preflight the required baseline before producing any measurements. An
    # unavailable cpp_boundary is a declined benchmark, never a Python fallback.
    baseline_preflight = _new_scalar()
    _assert_initial_state(
        baseline_preflight.snapshot(), label=f"{BASELINE_BACKEND} preflight state"
    )
    del baseline_preflight

    rows: dict[str, Any] = {}
    for batch_size in BATCH_SIZES:
        trace = trace_for_size(batch_size)
        packed_trace = _packed(trace)
        expected_state = _validate(trace, packed_trace)
        oracle = _new_scalar()
        expected_results = _scalar_results(oracle, trace)
        if oracle.snapshot() != expected_state:
            raise AssertionError("canonical result oracle changed the expected state")
        iterations = max(20, target_operations // batch_size)
        batch_samples: list[int] = []
        scalar_samples: list[int] = []
        for sample in range(samples):
            exact = _new_exact()
            scalar = _new_scalar()

            def time_exact() -> tuple[int, Any]:
                last_result = None
                started = time.perf_counter_ns()
                for _ in range(iterations):
                    result = exact.mutate_packed(packed_trace)
                    if last_result is not None:
                        del last_result
                    last_result = result
                elapsed = time.perf_counter_ns() - started
                if last_result is None:
                    raise AssertionError("timed exact loop produced no result")
                return elapsed, last_result

            def time_scalar() -> int:
                started = time.perf_counter_ns()
                for _ in range(iterations):
                    _scalar_once(scalar, trace)
                return time.perf_counter_ns() - started

            if sample % 2:
                scalar_elapsed = time_scalar()
                batch_elapsed, timed_packed_result = time_exact()
            else:
                batch_elapsed, timed_packed_result = time_exact()
                scalar_elapsed = time_scalar()
            timed_exact_results = timed_packed_result.materialize()
            timed_scalar_results = _scalar_results(scalar, trace)
            if timed_exact_results != expected_results:
                raise AssertionError("timed packed per-row results are not canonical")
            if timed_scalar_results != expected_results:
                raise AssertionError("timed scalar instance results are not canonical")
            if (
                exact.snapshot() != expected_state
                or scalar.snapshot() != expected_state
            ):
                raise AssertionError(
                    "repeated timed trace changed the 64-interval state"
                )
            logical_operations = iterations * batch_size
            batch_samples.append(batch_elapsed // logical_operations)
            scalar_samples.append(scalar_elapsed // logical_operations)

        comparison = paired_statistics(
            scalar_samples,
            batch_samples,
            ratio_limit=1.0,
            seed=PAIRED_BOOTSTRAP_SEED,
        )
        throughput_samples = [1_000_000_000 / value for value in batch_samples]
        throughput_ci = _median_confidence(throughput_samples)
        rows[str(batch_size)] = {
            "batch_size": batch_size,
            "classification": (
                "single-call-no-op-diagnostic" if batch_size == 1 else "restorative"
            ),
            "baseline_backend": BASELINE_BACKEND,
            "logical_operations_per_sample": iterations * batch_size,
            "batch_ns_per_operation_samples": batch_samples,
            "scalar_ns_per_operation_samples": scalar_samples,
            "batch_median_ops_per_second": statistics.median(throughput_samples),
            "batch_ops_per_second_confidence_95": throughput_ci,
            "paired": comparison,
            "speedup_confidence_95": (
                1.0 / comparison["confidence_95_ratio"][1],
                1.0 / comparison["confidence_95_ratio"][0],
            ),
        }

    materialize_source = _new_exact().mutate_packed(_packed(trace_for_size(16)))
    materialize_samples: list[int] = []
    materialize_iterations = max(100, target_operations // 16)
    for _ in range(samples):
        started = time.perf_counter_ns()
        for _ in range(materialize_iterations):
            result = materialize_source.materialize()
            del result
        elapsed = time.perf_counter_ns() - started
        materialize_samples.append(elapsed // (materialize_iterations * 16))

    return _build_report(
        rows=rows,
        materialize_samples=materialize_samples,
        samples=samples,
        target_operations=target_operations,
    )


def render_markdown(report: dict[str, Any], digest: str) -> str:
    """Render the concise Markdown companion bound to canonical JSON bytes."""
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


def write_artifacts(report: dict[str, Any], output: Path) -> tuple[Path, Path, Path]:
    """Atomically write canonical JSON, Markdown, and SHA-256 sidecar."""
    if output.suffix != ".json":
        raise ValueError("exact-batch output must use a .json suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    markdown = output.with_suffix(".md")
    checksum = Path(f"{output}.sha256")
    files: tuple[tuple[Path, bytes], ...] = (
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
    parser.add_argument("--target-operations", type=int, default=20_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--enforce-hard-gates",
        action="store_true",
        help="return failure if any locally callable exact-batch gate fails",
    )
    args = parser.parse_args()
    try:
        report = run_benchmark(
            samples=args.samples, target_operations=args.target_operations
        )
    except BackendUnavailableError as error:
        print(
            f"declined: required baseline backend {BASELINE_BACKEND!r} is unavailable: {error}",
            file=sys.stderr,
        )
        return 2
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        write_artifacts(report, args.output)
    print(encoded)
    gate_failed = args.enforce_hard_gates and not all(
        gate["passed"] for gate in report["gates"].values()
    )
    return 1 if gate_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
