#!/usr/bin/env python3
"""Canonical hot-path benchmark for the ``cpp_boundary`` range-set surface.

It compares, on a single validated ``cpp_boundary`` instance and workload, the
first-class mutation interfaces the range set exposes:

* ``add``/``discard`` returning ``MutationResult`` (synchronized and
  unsynchronized locking levels);
* the fully-native scalar ``release``/``reserve`` mutators (synchronized and
  unsynchronized);
* the plain-native floor calling the raw ``IntervalManager`` scalar mutators
  directly, with no range-set wrapper.

This is not a universal performance claim.  It reports one interface family, one
deterministic restorative workload, one host, one timing layer, and explicit
exclusions.  Manager construction, setup mutations, and every correctness check
run outside the timed region.
"""

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
from pathlib import Path
from typing import Any

from treemendous import BackendUnavailableError, Span, create_range_set

SCHEMA = "treemendous-rangeset-hotpath-benchmark-v1"
WORKLOAD_SCHEMA = "treemendous-rangeset-hotpath-workload-v1"
BACKEND = "cpp_boundary"
DOMAIN = (0, 4_096)
INITIAL_BLOCK_COUNT = 64
BLOCK_STRIDE = 32
FREE_BLOCK_LENGTH = 16
TRACE_BLOCKS = 16
THROUGHPUT_BOOTSTRAP_SEED = 16
BOOTSTRAP_RESAMPLES = 10_000
PATHS = (
    "mutation_result_synchronized",
    "mutation_result_unsynchronized",
    "scalar_synchronized",
    "scalar_unsynchronized",
    "native_floor",
)
# Opcodes: 0 add/release, 1 discard/reserve, 2 discard/reserve require_covered.
OP_ADD = 0
OP_DISCARD = 1
OP_DISCARD_REQUIRE_COVERED = 2
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

Operation = tuple[int, int, int]


def _initial_intervals() -> tuple[tuple[int, int], ...]:
    return tuple(
        (index * BLOCK_STRIDE, index * BLOCK_STRIDE + FREE_BLOCK_LENGTH)
        for index in range(INITIAL_BLOCK_COUNT)
    )


def _trace() -> tuple[Operation, ...]:
    """Return a deterministic, exactly restorative operation trace.

    Each block performs one real split-and-restore pair, one require-covered
    rejection (the tail crosses into an occupied gap), and one no-op add over an
    already-free block, so the net effect over the whole trace is the identity.
    """
    operations: list[Operation] = []
    for index in range(TRACE_BLOCKS):
        base = index * BLOCK_STRIDE
        operations.append((OP_DISCARD, base + 4, base + 8))  # real split
        operations.append((OP_ADD, base + 4, base + 8))  # exact restoration
        operations.append(
            (OP_DISCARD_REQUIRE_COVERED, base + 12, base + 20)  # rejected no-op
        )
        operations.append((OP_ADD, base, base + FREE_BLOCK_LENGTH))  # no-op add
    return tuple(operations)


_TRACE = _trace()


def _expected_changed_lengths() -> tuple[int, ...]:
    """Canonical per-operation ``changed_length`` for the restorative trace."""
    oracle = _new_range_set(synchronized=True)
    lengths: list[int] = []
    for opcode, start, end in _TRACE:
        span = Span(start, end)
        if opcode == OP_ADD:
            lengths.append(oracle.add(span).changed_length)
        else:
            lengths.append(
                oracle.discard(
                    span, require_covered=opcode == OP_DISCARD_REQUIRE_COVERED
                ).changed_length
            )
    if oracle.snapshot() != _canonical_initial_snapshot():
        raise AssertionError("benchmark trace is not exactly restorative")
    return tuple(lengths)


def _new_range_set(*, synchronized: bool) -> Any:
    manager = create_range_set(
        DOMAIN, backend=BACKEND, initially_available=False, synchronized=synchronized
    )
    for start, end in _initial_intervals():
        manager.add(Span(start, end))
    return manager


def _new_native_floor() -> Any:
    module = importlib.import_module("treemendous.cpp.boundary")
    manager = module.IntervalManager()
    manager.set_managed_domain([list(DOMAIN)])
    for start, end in _initial_intervals():
        manager.release_interval(start, end)
    return manager


def _canonical_initial_snapshot() -> Any:
    reference = _new_range_set(synchronized=True)
    return reference.snapshot()


# Spans and packed native tuples are built once, outside the timed region, so
# each measurement isolates the mutation-call cost of one interface rather than
# per-operation ``Span`` validation or tuple unpacking done by the caller.
_SPAN_TRACE: tuple[tuple[int, Span, bool], ...] = tuple(
    (opcode, Span(start, end), opcode == OP_DISCARD_REQUIRE_COVERED)
    for opcode, start, end in _TRACE
)
_NATIVE_TRACE: tuple[tuple[int, int, int, bool], ...] = tuple(
    (opcode, start, end, opcode == OP_DISCARD_REQUIRE_COVERED)
    for opcode, start, end in _TRACE
)


def _exec_mutation_result(manager: Any) -> None:
    for opcode, span, require_covered in _SPAN_TRACE:
        if opcode == OP_ADD:
            manager.add(span)
        else:
            manager.discard(span, require_covered=require_covered)


def _exec_scalar(manager: Any) -> None:
    for opcode, span, require_covered in _SPAN_TRACE:
        if opcode == OP_ADD:
            manager.release(span)
        else:
            manager.reserve(span, require_covered=require_covered)


def _exec_native_floor(manager: Any) -> None:
    for opcode, start, end, require_covered in _NATIVE_TRACE:
        if opcode == OP_ADD:
            manager.release_delta_length(start, end)
        else:
            manager.reserve_delta_length(start, end, require_covered)


def _collect_mutation_result(manager: Any) -> list[int]:
    lengths: list[int] = []
    for opcode, span, require_covered in _SPAN_TRACE:
        if opcode == OP_ADD:
            lengths.append(manager.add(span).changed_length)
        else:
            lengths.append(
                manager.discard(span, require_covered=require_covered).changed_length
            )
    return lengths


def _collect_scalar(manager: Any) -> list[int]:
    lengths: list[int] = []
    for opcode, span, require_covered in _SPAN_TRACE:
        if opcode == OP_ADD:
            lengths.append(manager.release(span))
        else:
            lengths.append(manager.reserve(span, require_covered=require_covered))
    return lengths


def _collect_native_floor(manager: Any) -> list[int]:
    lengths: list[int] = []
    for opcode, start, end, require_covered in _NATIVE_TRACE:
        if opcode == OP_ADD:
            lengths.append(manager.release_delta_length(start, end))
        else:
            lengths.append(manager.reserve_delta_length(start, end, require_covered))
    return lengths


_PATH_RUNNERS = {
    "mutation_result_synchronized": (
        lambda: _new_range_set(synchronized=True),
        _exec_mutation_result,
        _collect_mutation_result,
        "RangeSet.add/RangeSet.discard (MutationResult), synchronized RLock",
    ),
    "mutation_result_unsynchronized": (
        lambda: _new_range_set(synchronized=False),
        _exec_mutation_result,
        _collect_mutation_result,
        "RangeSet.add/RangeSet.discard (MutationResult), unsynchronized no-op lock",
    ),
    "scalar_synchronized": (
        lambda: _new_range_set(synchronized=True),
        _exec_scalar,
        _collect_scalar,
        "RangeSet.release/RangeSet.reserve (scalar), synchronized RLock",
    ),
    "scalar_unsynchronized": (
        lambda: _new_range_set(synchronized=False),
        _exec_scalar,
        _collect_scalar,
        "RangeSet.release/RangeSet.reserve (scalar), unsynchronized no-op lock",
    ),
    "native_floor": (
        _new_native_floor,
        _exec_native_floor,
        _collect_native_floor,
        "raw IntervalManager.release_delta_length/reserve_delta_length, no wrapper",
    ),
}


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


def workload_manifest() -> dict[str, Any]:
    body = {
        "schema": WORKLOAD_SCHEMA,
        "domain": list(DOMAIN),
        "initial_intervals": [list(pair) for pair in _initial_intervals()],
        "operations": [list(operation) for operation in _TRACE],
        "expected_changed_lengths": list(_expected_changed_lengths()),
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
            [compiler, "--version"], check=True, capture_output=True, text=True
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
        "extensions": {BACKEND: _extension_metadata("treemendous.cpp.boundary")},
    }
    return candidate, environment, build


def _validate_path(name: str, expected_lengths: tuple[int, ...]) -> None:
    """Confirm one path's per-op deltas and restorative final state (untimed)."""
    factory, _executor, collector, _interface = _PATH_RUNNERS[name]
    manager = factory()
    lengths = tuple(collector(manager))
    if lengths != expected_lengths:
        raise AssertionError(f"path {name!r} produced non-canonical changed lengths")
    if name == "native_floor":
        observed = tuple((start, end) for start, end in manager.get_intervals())
        if observed != _initial_intervals():
            raise AssertionError(f"path {name!r} did not restore the initial state")
        return
    if manager.snapshot() != _canonical_initial_snapshot():
        raise AssertionError(f"path {name!r} did not restore the initial state")


def run_benchmark(
    *, samples: int = 30, target_operations: int = 20_000
) -> dict[str, Any]:
    if samples < 20:
        raise ValueError("hot-path benchmark requires at least 20 samples")
    if target_operations < 1:
        raise ValueError("target_operations must be positive")

    # Preflight the required backend; an unavailable cpp_boundary is a declined
    # benchmark, never a silent Python fallback.
    _new_range_set(synchronized=True)

    expected_lengths = _expected_changed_lengths()
    operations_per_trace = len(_TRACE)
    iterations = max(20, target_operations // operations_per_trace)

    paths: dict[str, Any] = {}
    for name in PATHS:
        _validate_path(name, expected_lengths)
        factory, executor, _collector, interface = _PATH_RUNNERS[name]
        ns_samples: list[int] = []
        for _ in range(samples):
            manager = factory()  # construction/setup excluded from timing
            started = time.perf_counter_ns()
            for _ in range(iterations):
                executor(manager)
            elapsed = time.perf_counter_ns() - started
            # Restorative trace: the timed instance returns to the initial state.
            ns_samples.append(elapsed // (iterations * operations_per_trace))
        throughput = [1_000_000_000 / value for value in ns_samples]
        paths[name] = {
            "interface": interface,
            "operations_per_trace": operations_per_trace,
            "logical_operations_per_sample": iterations * operations_per_trace,
            "ns_per_operation_samples": ns_samples,
            "median_ns_per_operation": statistics.median(ns_samples),
            "median_ops_per_second": statistics.median(throughput),
            "ops_per_second_confidence_95": list(_median_confidence(throughput)),
        }

    candidate, environment, build = _provenance()
    return {
        "schema": SCHEMA,
        "candidate": candidate,
        "environment": environment,
        "build": build,
        "backend": BACKEND,
        "methodology": {
            "samples": samples,
            "target_operations": target_operations,
            "workload": (
                "deterministic 64-operation restorative trace over 64 free "
                "blocks; net effect is the identity"
            ),
            "timed_layer": (
                "only the public mutation calls of each interface; the trace "
                "runs in-process without per-operation Python bookkeeping"
            ),
            "excluded": (
                "manager/setup construction, initial releases, canonical delta "
                "and restorative-state validation, and snapshot reads"
            ),
            "throughput_bootstrap_seed": THROUGHPUT_BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "paths": list(PATHS),
            "universal_claim": False,
        },
        "workload_manifest": workload_manifest(),
        "paths": paths,
    }


def render_markdown(report: dict[str, Any], digest: str) -> str:
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


def write_artifacts(report: dict[str, Any], output: Path) -> tuple[Path, Path, Path]:
    if output.suffix != ".json":
        raise ValueError("hot-path output must use a .json suffix")
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
    args = parser.parse_args()
    try:
        report = run_benchmark(
            samples=args.samples, target_operations=args.target_operations
        )
    except BackendUnavailableError as error:
        print(
            f"declined: required backend {BACKEND!r} is unavailable: {error}",
            file=sys.stderr,
        )
        return 2
    if args.output:
        write_artifacts(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
