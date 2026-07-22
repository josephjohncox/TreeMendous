#!/usr/bin/env python3
"""Correctness-attested ExactBatch application-shape diagnostic.

This experiment compares the packed native call with canonical ``cpp_boundary``
scalar replay. It does not alter or replace any stable exact-batch gate and does
not define a universal break-even threshold.
"""

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
import struct
import subprocess
import sysconfig
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from treemendous import Span, create_range_set
from treemendous.exact_batch import (
    BatchLimits,
    ExactBatchRangeSet,
    MutationOpcode,
)

SCHEMA = "treemendous-exact-batch-application-matrix-v1"
WORKLOAD_SCHEMA = "treemendous-exact-batch-application-workload-v1"
BASELINE_BACKEND = "cpp_boundary"
LOCAL_INTERVAL_COUNTS = (64, 1_000, 10_000)
CLI_INTERVAL_COUNT = 100_000
BATCH_SIZES = (1, 4, 16, 64)
LOCALITIES = ("head", "middle", "tail")
SHAPES = (
    "strict_accept_reject",
    "idempotent_real_noop",
    "fragment_restore",
    "coalesce_restore",
    "wide_fanout",
)
MINIMUM_SAMPLES = 10
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 841_064
BUILD_FLAG_NAMES = (
    "BOOST_ROOT",
    "TREE_MENDOUS_DISABLE_OPTIMIZATIONS",
    "TREE_MENDOUS_GLIBCXX_DEBUG",
    "TREE_MENDOUS_LOCAL_NATIVE",
    "TREE_MENDOUS_SANITIZERS",
    "TREE_MENDOUS_WITH_ICL",
)
RESOURCE_LIMITS = {
    "max_operations": 64,
    "max_live_intervals": 100_005,
    "max_changed_spans": 1_000_000,
    "max_result_bytes": 64 * 1024 * 1024,
    "max_work_units": 10_000_000,
}
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_OPERATION = struct.Struct("@qqq")

Operation = tuple[int, int, int]


@dataclass(frozen=True)
class CaseDefinition:
    """One deterministic matrix case, including untimed state setup."""

    case_id: str
    interval_count: int
    batch_size: int
    locality: str
    shape: str
    domain: tuple[tuple[int, int], ...]
    setup: tuple[Operation, ...]
    operations: tuple[Operation, ...]
    fanout: int


def _checksum(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"diagnostic JSON contains duplicate key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"diagnostic JSON contains non-finite number: {value}")


def _validate_finite_numbers(value: Any) -> None:
    if type(value) is float and not math.isfinite(value):
        raise ValueError("diagnostic JSON contains non-finite number")
    if isinstance(value, dict):
        for item in value.values():
            _validate_finite_numbers(item)
    elif isinstance(value, list):
        for item in value:
            _validate_finite_numbers(item)


def _json_exact_equal(actual: Any, expected: Any) -> bool:
    """Compare reconstructed JSON values without Python numeric coercion."""
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(
            _json_exact_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _json_exact_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _components(count: int) -> tuple[tuple[int, int], ...]:
    origin = -4 * count
    return tuple((origin + 8 * index, origin + 8 * index + 6) for index in range(count))


def _wide_components(
    interval_count: int, locality: str
) -> tuple[tuple[tuple[int, int], ...], tuple[Operation, ...], tuple[int, int], int]:
    fanout = min(8, interval_count)
    component_count = interval_count - fanout + 1
    wide_index = _local_index(component_count, locality)
    cursor = -4 * interval_count
    domain: list[tuple[int, int]] = []
    setup: list[Operation] = []
    wide_span = (0, 0)
    for index in range(component_count):
        if index == wide_index:
            lower = cursor
            upper = lower + 2 * fanout - 1
            domain.append((lower, upper))
            setup.extend(
                (
                    MutationOpcode.DISCARD_REQUIRE_COVERED,
                    lower + offset,
                    lower + offset + 1,
                )
                for offset in range(1, 2 * fanout - 1, 2)
            )
            wide_span = (lower, upper)
            cursor = upper + 2
        else:
            domain.append((cursor, cursor + 6))
            cursor += 8
    return tuple(domain), tuple(setup), wide_span, fanout


def _local_index(count: int, locality: str) -> int:
    if locality == "head":
        return 0
    if locality == "middle":
        return count // 2
    if locality == "tail":
        return count - 1
    raise ValueError(f"unsupported locality: {locality}")


def _repeat_cycle(
    cycle: tuple[Operation, ...], batch_size: int
) -> tuple[Operation, ...]:
    return tuple(cycle[index % len(cycle)] for index in range(batch_size))


def case_definition(
    interval_count: int,
    batch_size: int,
    shape: str,
    locality: str,
) -> CaseDefinition:
    """Generate one signed-coordinate, half-open application-shaped trace."""
    if interval_count < 4:
        raise ValueError("interval_count must be at least four")
    if batch_size not in BATCH_SIZES:
        raise ValueError(f"unsupported batch size: {batch_size}")
    if shape not in SHAPES:
        raise ValueError(f"unsupported shape: {shape}")
    if locality not in LOCALITIES:
        raise ValueError(f"unsupported locality: {locality}")

    component_count = (
        interval_count - 1 if shape == "coalesce_restore" else interval_count
    )
    domain = _components(component_count)
    target_index = _local_index(component_count, locality)
    lower, upper = domain[target_index]
    setup: tuple[Operation, ...] = ()
    fanout = 1

    if shape == "strict_accept_reject":
        span = (lower + 2, lower + 4)
        cycle = (
            (MutationOpcode.DISCARD_REQUIRE_COVERED, *span),
            (MutationOpcode.DISCARD_REQUIRE_COVERED, *span),
            (MutationOpcode.ADD, *span),
            (MutationOpcode.ADD, *span),
        )
    elif shape == "idempotent_real_noop":
        span = (lower + 2, lower + 4)
        cycle = (
            (MutationOpcode.DISCARD, *span),
            (MutationOpcode.DISCARD, *span),
            (MutationOpcode.ADD, *span),
            (MutationOpcode.ADD, *span),
        )
    elif shape == "fragment_restore":
        left = (lower + 1, lower + 2)
        right = (upper - 2, upper - 1)
        cycle = (
            (MutationOpcode.DISCARD, *left),
            (MutationOpcode.ADD, *left),
            (MutationOpcode.DISCARD_REQUIRE_COVERED, *right),
            (MutationOpcode.ADD, *right),
        )
    elif shape == "coalesce_restore":
        gap = (lower + 2, lower + 4)
        setup = ((MutationOpcode.DISCARD_REQUIRE_COVERED, *gap),)
        cycle = (
            (MutationOpcode.ADD, *gap),
            (MutationOpcode.DISCARD_REQUIRE_COVERED, *gap),
            (MutationOpcode.ADD, *gap),
            (MutationOpcode.DISCARD_REQUIRE_COVERED, *gap),
        )
    else:
        domain, setup, wide_span, fanout = _wide_components(interval_count, locality)
        start, end = wide_span
        cycle = (
            (MutationOpcode.DISCARD, start, end),
            (MutationOpcode.ADD, start, end),
            (MutationOpcode.DISCARD_REQUIRE_COVERED, start, end),
            (MutationOpcode.ADD, start, end),
        )

    operations = _repeat_cycle(cycle, batch_size)
    case_id = f"n{interval_count}-b{batch_size}-{shape}-{locality}"
    return CaseDefinition(
        case_id,
        interval_count,
        batch_size,
        locality,
        shape,
        domain,
        setup,
        operations,
        fanout,
    )


def _packed(operations: tuple[Operation, ...]) -> bytes:
    return b"".join(
        _OPERATION.pack(int(opcode), start, end) for opcode, start, end in operations
    )


def _apply_scalar(manager: Any, operations: tuple[Operation, ...]) -> tuple[Any, ...]:
    results = []
    for opcode, start, end in operations:
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


def _geometry(snapshot: Any) -> tuple[tuple[int, int], ...]:
    return tuple((interval.start, interval.end) for interval in snapshot.intervals)


def _serialized_results(results: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [
        {
            "changed": [[span.start, span.end] for span in result.changed],
            "changed_length": result.changed_length,
            "fully_covered": result.fully_covered,
        }
        for result in results
    ]


def _new_scalar(case: CaseDefinition) -> Any:
    manager = create_range_set(
        case.domain,
        backend=BASELINE_BACKEND,
        initially_available=True,
    )
    _apply_scalar(manager, case.setup)
    return manager


def _new_exact(case: CaseDefinition) -> ExactBatchRangeSet:
    limits = BatchLimits(**RESOURCE_LIMITS)
    if asdict(limits) != RESOURCE_LIMITS:
        raise AssertionError("diagnostic limits no longer match BatchLimits")
    manager = ExactBatchRangeSet(case.domain, initially_available=True, limits=limits)
    if case.setup:
        manager.mutate_packed(_packed(case.setup)).materialize()
    return manager


def _oracle_evidence(
    case: CaseDefinition,
) -> tuple[tuple[Any, ...], tuple[tuple[int, int], ...], dict[str, int]]:
    manager = _new_scalar(case)
    if len(manager.snapshot().intervals) != case.interval_count:
        raise AssertionError(f"{case.case_id}: setup has wrong live interval count")
    work_units = 0
    results: list[Any] = []
    for operation in case.operations:
        work_units += 1 + len(manager.snapshot().intervals)
        results.extend(_apply_scalar(manager, (operation,)))
    result_tuple = tuple(results)
    final_geometry = _geometry(manager.snapshot())
    declaration = {
        "logical_rows": case.batch_size,
        "initial_live_intervals": case.interval_count,
        "final_live_intervals": len(final_geometry),
        "changed_spans": sum(len(result.changed) for result in result_tuple),
        "changed_length": sum(result.changed_length for result in result_tuple),
        "mutating_rows": sum(bool(result.changed) for result in result_tuple),
        "noop_rows": sum(not result.changed for result in result_tuple),
        "strict_accepted_rows": sum(
            operation[0] == MutationOpcode.DISCARD_REQUIRE_COVERED
            and result.fully_covered
            for operation, result in zip(case.operations, result_tuple, strict=True)
        ),
        "strict_rejected_rows": sum(
            operation[0] == MutationOpcode.DISCARD_REQUIRE_COVERED
            and not result.fully_covered
            for operation, result in zip(case.operations, result_tuple, strict=True)
        ),
        "native_work_units": work_units,
        "fanout": case.fanout,
    }
    return result_tuple, final_geometry, declaration


def _case_manifest(case: CaseDefinition) -> dict[str, Any]:
    results, final_geometry, work = _oracle_evidence(case)
    oracle = {
        "results": _serialized_results(results),
        "final_geometry": [list(span) for span in final_geometry],
    }
    body = {
        "case_id": case.case_id,
        "interval_count": case.interval_count,
        "batch_size": case.batch_size,
        "shape": case.shape,
        "locality": case.locality,
        "domain_generator": (
            "one wide component fragmented by unit setup holes"
            if case.shape == "wide_fanout"
            else "component_i=[-4*C+8*i,-4*C+8*i+6)"
        ),
        "domain_components": len(case.domain),
        "setup_operations": [list(map(int, row)) for row in case.setup],
        "operations": [list(map(int, row)) for row in case.operations],
        "packed_input_bytes": len(_packed(case.operations)),
        "work": work,
        "oracle_digest": _checksum(oracle),
    }
    return {**body, "digest": _checksum(body)}


def workload_manifest(cases: tuple[CaseDefinition, ...]) -> dict[str, Any]:
    """Return the exact generated workload plus a whole-manifest digest."""
    body = {
        "schema": WORKLOAD_SCHEMA,
        "coordinate_semantics": "signed-int64 half-open integer intervals",
        "cases": [_case_manifest(case) for case in cases],
    }
    return {**body, "digest": _checksum(body)}


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


def attest_case(
    case: CaseDefinition,
) -> tuple[tuple[Any, ...], tuple[tuple[int, int], ...], int, dict[str, int]]:
    """Prove packed rows and final state against canonical scalar replay."""
    expected_rows, expected_final, work = _oracle_evidence(case)
    exact = _new_exact(case)
    initial = exact.snapshot()
    if len(initial.intervals) != case.interval_count:
        raise AssertionError(f"{case.case_id}: exact setup has wrong state count")
    packed_result = exact.mutate_packed(_packed(case.operations))
    exact_rows = packed_result.materialize()
    exact_final = _geometry(exact.snapshot())
    if exact_rows != expected_rows:
        raise AssertionError(f"{case.case_id}: packed rows differ from cpp_boundary")
    if exact_final != expected_final:
        raise AssertionError(f"{case.case_id}: final state differs from cpp_boundary")
    return expected_rows, expected_final, _packed_result_bytes(packed_result), work


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _median_confidence(values: list[float]) -> list[float]:
    rng = random.Random(BOOTSTRAP_SEED)
    bootstrapped = [
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return [_percentile(bootstrapped, 0.025), _percentile(bootstrapped, 0.975)]


def _measure_case(case: CaseDefinition, samples: int) -> dict[str, Any]:
    expected_rows, expected_final, packed_bytes, work = attest_case(case)
    packed_operations = _packed(case.operations)
    exact_samples: list[int] = []
    scalar_samples: list[int] = []

    for sample_index in range(samples):
        exact = _new_exact(case)
        scalar = _new_scalar(case)
        if sample_index % 2 == 0:
            started = time.perf_counter_ns()
            packed_result = exact.mutate_packed(packed_operations)
            exact_elapsed = time.perf_counter_ns() - started
            started = time.perf_counter_ns()
            scalar_rows = _apply_scalar(scalar, case.operations)
            scalar_elapsed = time.perf_counter_ns() - started
        else:
            started = time.perf_counter_ns()
            scalar_rows = _apply_scalar(scalar, case.operations)
            scalar_elapsed = time.perf_counter_ns() - started
            started = time.perf_counter_ns()
            packed_result = exact.mutate_packed(packed_operations)
            exact_elapsed = time.perf_counter_ns() - started

        exact_rows = packed_result.materialize()
        if exact_rows != expected_rows or scalar_rows != expected_rows:
            raise AssertionError(f"{case.case_id}: timed result diverged from oracle")
        if _geometry(exact.snapshot()) != expected_final:
            raise AssertionError(f"{case.case_id}: timed packed final state diverged")
        if _geometry(scalar.snapshot()) != expected_final:
            raise AssertionError(f"{case.case_id}: timed scalar final state diverged")
        if _packed_result_bytes(packed_result) != packed_bytes:
            raise AssertionError(f"{case.case_id}: packed result byte count changed")
        exact_samples.append(exact_elapsed)
        scalar_samples.append(scalar_elapsed)

    ratios = [
        exact / scalar
        for exact, scalar in zip(exact_samples, scalar_samples, strict=True)
    ]
    return {
        "case_id": case.case_id,
        "interval_count": case.interval_count,
        "batch_size": case.batch_size,
        "shape": case.shape,
        "locality": case.locality,
        "exact_latency_ns_samples": exact_samples,
        "exact_latency_ns_median": statistics.median(exact_samples),
        "exact_latency_ns_confidence_95": _median_confidence(
            [float(value) for value in exact_samples]
        ),
        "scalar_latency_ns_samples": scalar_samples,
        "scalar_latency_ns_median": statistics.median(scalar_samples),
        "scalar_latency_ns_confidence_95": _median_confidence(
            [float(value) for value in scalar_samples]
        ),
        "paired_exact_over_scalar_samples": ratios,
        "paired_exact_over_scalar_median": statistics.median(ratios),
        "paired_exact_over_scalar_confidence_95": _median_confidence(ratios),
        "packed_input_bytes": len(packed_operations),
        "packed_result_bytes": packed_bytes,
        "work": work,
        "validated_sample_count": samples,
    }


def _git_metadata() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        head_tree = git("rev-parse", "HEAD^{tree}").strip()
        status = git("status", "--porcelain=v1")
        diff = git("diff", "--binary", "HEAD")
        staged = [
            line for line in git("diff", "--cached", "--name-only").splitlines() if line
        ]
        changed_paths = sorted(
            line
            for line in git(
                "ls-files", "--modified", "--others", "--exclude-standard"
            ).splitlines()
            if line
        )
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": "unknown",
            "head_tree": "unknown",
            "clean_worktree": False,
            "changed_paths": [],
            "staged_files": [],
            "source_state_sha256": "unknown",
        }
    source_state = hashlib.sha256((status + diff).encode())
    for relative in changed_paths:
        path = _REPOSITORY_ROOT / relative
        source_state.update(relative.encode())
        source_state.update(b"\0")
        if path.is_file():
            source_state.update(hashlib.sha256(path.read_bytes()).digest())
    return {
        "commit": commit,
        "head_tree": head_tree,
        "clean_worktree": not status,
        "changed_paths": changed_paths,
        "staged_files": staged,
        "source_state_sha256": source_state.hexdigest(),
    }


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


def _source_metadata(relative: str) -> dict[str, str]:
    path = (_REPOSITORY_ROOT / relative).resolve()
    return {
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


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


def _runtime_metadata() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or 0,
    }


def _build_metadata() -> dict[str, Any]:
    return {
        "command": os.environ.get(
            "TREE_MENDOUS_BUILD_COMMAND", "python setup.py build_ext --inplace"
        ),
        "cxx": os.environ.get("CXX", "c++"),
        "cxx_version": _compiler_version(),
        "cc": str(sysconfig.get_config_var("CC") or "unknown"),
        "cflags": str(sysconfig.get_config_var("CFLAGS") or "unknown"),
        "flags": {name: os.environ.get(name, "") for name in BUILD_FLAG_NAMES},
    }


def _backend_metadata() -> dict[str, dict[str, Any]]:
    exact = ExactBatchRangeSet((0, 1), initially_available=False)
    exact_type = type(exact._manager)
    baseline = create_range_set(
        (0, 1), backend=BASELINE_BACKEND, initially_available=False
    )
    baseline_type = type(baseline._adapter.implementation)
    return {
        "exact_batch": {
            "id": "exact_batch",
            "module": exact_type.__module__,
            "type": exact_type.__qualname__,
            "extension": _extension_metadata("treemendous.cpp._exact_batch"),
        },
        BASELINE_BACKEND: {
            "id": BASELINE_BACKEND,
            "module": baseline_type.__module__,
            "type": baseline_type.__qualname__,
            "extension": _extension_metadata("treemendous.cpp.boundary"),
        },
    }


def _provenance() -> dict[str, Any]:
    return {
        "source": _git_metadata(),
        "sources": [
            _source_metadata(
                "tests/performance/experiments/exact_batch_application_matrix.py"
            ),
            _source_metadata("treemendous/exact_batch.py"),
        ],
        "runtime": _runtime_metadata(),
        "build": _build_metadata(),
        "backends": _backend_metadata(),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    break_even: dict[str, int | None] = {}
    for interval_count in sorted({row["interval_count"] for row in rows}):
        for shape in SHAPES:
            for locality in LOCALITIES:
                matching = sorted(
                    (
                        row
                        for row in rows
                        if row["interval_count"] == interval_count
                        and row["shape"] == shape
                        and row["locality"] == locality
                    ),
                    key=lambda row: row["batch_size"],
                )
                winner = next(
                    (
                        row["batch_size"]
                        for row in matching
                        if row["paired_exact_over_scalar_median"] <= 1.0
                    ),
                    None,
                )
                break_even[f"n{interval_count}-{shape}-{locality}"] = winner
    return {
        "observed_median_break_even_batch": break_even,
        "cells_exact_faster_with_95_ci": sum(
            row["paired_exact_over_scalar_confidence_95"][1] < 1.0 for row in rows
        ),
        "cells_scalar_faster_with_95_ci": sum(
            row["paired_exact_over_scalar_confidence_95"][0] > 1.0 for row in rows
        ),
        "cells_inconclusive": sum(
            row["paired_exact_over_scalar_confidence_95"][0]
            <= 1.0
            <= row["paired_exact_over_scalar_confidence_95"][1]
            for row in rows
        ),
    }


def _cases(
    interval_counts: tuple[int, ...],
    batch_sizes: tuple[int, ...],
    shapes: tuple[str, ...],
    localities: tuple[str, ...],
) -> tuple[CaseDefinition, ...]:
    return tuple(
        case_definition(interval_count, batch_size, shape, locality)
        for interval_count in interval_counts
        for batch_size in batch_sizes
        for shape in shapes
        for locality in localities
    )


def _run_matrix(
    *,
    profile: str,
    samples: int,
    interval_counts: tuple[int, ...],
    batch_sizes: tuple[int, ...],
    shapes: tuple[str, ...],
    localities: tuple[str, ...],
) -> dict[str, Any]:
    if samples < MINIMUM_SAMPLES:
        raise ValueError(f"diagnostic requires at least {MINIMUM_SAMPLES} samples")
    cases = _cases(interval_counts, batch_sizes, shapes, localities)
    manifest = workload_manifest(cases)
    rows = [_measure_case(case, samples) for case in cases]
    return {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "provenance": _provenance(),
        "methodology": {
            "profile": profile,
            "samples": samples,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "interval_counts": list(interval_counts),
            "batch_sizes": list(batch_sizes),
            "shapes": list(shapes),
            "localities": list(localities),
            "timed_exact_layer": "one mutate_packed call through packed-result construction",
            "timed_scalar_layer": "ordered cpp_boundary RangeSet add/discard replay",
            "excluded": "manager/domain setup, operation packing, scalar oracle construction, result materialization, snapshots, validation, and artifact writing",
            "pair_order": "alternated exact-first and scalar-first by sample index",
            "claim_scope": "bounded diagnostic cells only; no universal threshold or stable gate",
        },
        "resource_limits": dict(RESOURCE_LIMITS),
        "workload_manifest": manifest,
        "rows": rows,
        "summary": _summary(rows),
        "existing_exact_gates_changed": False,
    }


def run_benchmark(
    *,
    profile: str = "smoke",
    samples: int = MINIMUM_SAMPLES,
    include_100000: bool = False,
) -> dict[str, Any]:
    """Run a smoke or complete local diagnostic profile."""
    interval_counts: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    localities: tuple[str, ...]
    if profile == "smoke":
        interval_counts = (64,)
        batch_sizes = (1, 16)
        localities = ("head", "tail")
    elif profile == "local":
        interval_counts = LOCAL_INTERVAL_COUNTS
        batch_sizes = BATCH_SIZES
        localities = LOCALITIES
    else:
        raise ValueError(f"unsupported profile: {profile}")
    if include_100000:
        interval_counts = (*interval_counts, CLI_INTERVAL_COUNT)
    return _run_matrix(
        profile=profile,
        samples=samples,
        interval_counts=interval_counts,
        batch_sizes=batch_sizes,
        shapes=SHAPES,
        localities=localities,
    )


def render_markdown(report: dict[str, Any], digest: str) -> str:
    """Render a deterministic human-readable companion."""
    source = report["provenance"]["source"]
    lines = [
        "# Exact-batch application matrix (diagnostic)",
        "",
        "This bounded observation does not change stable exact-batch gates and is not a universal performance claim.",
        "",
        f"- Commit: `{source['commit']}`",
        f"- Clean worktree: `{str(source['clean_worktree']).lower()}`",
        f"- Workload digest: `{report['workload_manifest']['digest']}`",
        f"- JSON SHA-256: `{digest}`",
        f"- Samples per cell: `{report['methodology']['samples']}`",
        "",
        "| N | B | Shape | Locality | Packed median (us) | Scalar median (us) | Packed/scalar median | 95% paired CI | Result bytes | Work units |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        confidence = row["paired_exact_over_scalar_confidence_95"]
        lines.append(
            f"| {row['interval_count']} | {row['batch_size']} | {row['shape']} | "
            f"{row['locality']} | {row['exact_latency_ns_median'] / 1_000:.3f} | "
            f"{row['scalar_latency_ns_median'] / 1_000:.3f} | "
            f"{row['paired_exact_over_scalar_median']:.3f} | "
            f"[{confidence[0]:.3f}, {confidence[1]:.3f}] | "
            f"{row['packed_result_bytes']} | {row['work']['native_work_units']} |"
        )
    summary = report["summary"]
    lines.extend(
        (
            "",
            f"Cells with packed faster at 95% CI: {summary['cells_exact_faster_with_95_ci']}.",
            f"Cells with scalar faster at 95% CI: {summary['cells_scalar_faster_with_95_ci']}.",
            f"Inconclusive cells: {summary['cells_inconclusive']}.",
        )
    )
    return "\n".join(lines) + "\n"


def write_artifacts(report: dict[str, Any], output: Path) -> tuple[Path, Path, Path]:
    """Atomically write canonical JSON, Markdown, and SHA-256 sidecar."""
    if output.suffix != ".json":
        raise ValueError("diagnostic output must use a .json suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    markdown = output.with_suffix(".md")
    checksum = Path(f"{output}.sha256")
    for destination, contents in (
        (output, encoded),
        (markdown, render_markdown(report, digest).encode()),
        (checksum, f"{digest}  {output.name}\n".encode()),
    ):
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_bytes(contents)
        temporary.replace(destination)
    return output, markdown, checksum


def _require_exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"diagnostic {label} keys/type mismatch")
    return value


def _expected_profile_dimensions(methodology: dict[str, Any]) -> None:
    profile = methodology["profile"]
    include_large = methodology["interval_counts"][-1:] == [CLI_INTERVAL_COUNT]
    if profile == "smoke":
        counts = [64, *([CLI_INTERVAL_COUNT] if include_large else [])]
        batches = [1, 16]
        localities = ["head", "tail"]
    elif profile == "local":
        counts = [
            *LOCAL_INTERVAL_COUNTS,
            *([CLI_INTERVAL_COUNT] if include_large else []),
        ]
        batches = list(BATCH_SIZES)
        localities = list(LOCALITIES)
    else:
        raise ValueError("diagnostic methodology profile is invalid")
    if (
        methodology["interval_counts"] != counts
        or methodology["batch_sizes"] != batches
        or methodology["shapes"] != list(SHAPES)
        or methodology["localities"] != localities
    ):
        raise ValueError("diagnostic methodology matrix is invalid")


def _expected_packed_result_bytes(case: CaseDefinition) -> int:
    rows, _, work = _oracle_evidence(case)
    if work["changed_spans"] != sum(len(row.changed) for row in rows):
        raise AssertionError("oracle work declaration diverged")
    return (
        (case.batch_size + 1) * 8
        + work["changed_spans"] * 16
        + case.batch_size * 8
        + case.batch_size
    )


def _verify_provenance(value: Any) -> None:
    provenance = _require_exact_keys(
        value, {"source", "sources", "runtime", "build", "backends"}, "provenance"
    )
    source = _require_exact_keys(
        provenance["source"],
        {
            "commit",
            "head_tree",
            "clean_worktree",
            "changed_paths",
            "staged_files",
            "source_state_sha256",
        },
        "source provenance",
    )
    if not _json_exact_equal(source, _git_metadata()):
        raise ValueError("diagnostic source provenance does not match local checkout")
    expected_sources = [
        _source_metadata(
            "tests/performance/experiments/exact_batch_application_matrix.py"
        ),
        _source_metadata("treemendous/exact_batch.py"),
    ]
    if not _json_exact_equal(provenance["sources"], expected_sources):
        raise ValueError("diagnostic source file path/hash mismatch")
    if not _json_exact_equal(provenance["runtime"], _runtime_metadata()):
        raise ValueError("diagnostic runtime provenance does not match current runtime")
    if not _json_exact_equal(provenance["build"], _build_metadata()):
        raise ValueError("diagnostic build provenance does not match current build")
    if not _json_exact_equal(provenance["backends"], _backend_metadata()):
        raise ValueError(
            "diagnostic backend/extension provenance does not match active backends"
        )


def verify_artifacts(output: Path) -> dict[str, Any]:
    """Verify the diagnostic triplet, provenance, workload, and every derivation."""
    markdown = output.with_suffix(".md")
    checksum = Path(f"{output}.sha256")
    for path in (output, markdown, checksum):
        if not path.is_file():
            raise ValueError(f"diagnostic artifact is missing: {path}")
    encoded = output.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if checksum.read_text() != f"{digest}  {output.name}\n":
        raise ValueError("diagnostic checksum does not match JSON")
    report = json.loads(
        encoded,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_non_finite,
    )
    if not isinstance(report, dict):
        raise ValueError("diagnostic JSON root must be an object")
    _validate_finite_numbers(report)
    if encoded != (json.dumps(report, indent=2, sort_keys=True) + "\n").encode():
        raise ValueError("diagnostic JSON is not canonical")
    _require_exact_keys(
        report,
        {
            "schema",
            "diagnostic_only",
            "provenance",
            "methodology",
            "resource_limits",
            "workload_manifest",
            "rows",
            "summary",
            "existing_exact_gates_changed",
        },
        "report",
    )
    if report["schema"] != SCHEMA or report["diagnostic_only"] is not True:
        raise ValueError("unexpected diagnostic schema")
    if report["existing_exact_gates_changed"] is not False:
        raise ValueError("existing exact gate declaration changed")
    if not _json_exact_equal(report["resource_limits"], RESOURCE_LIMITS):
        raise ValueError("diagnostic resource limits do not match")
    _verify_provenance(report["provenance"])

    methodology = _require_exact_keys(
        report["methodology"],
        {
            "profile",
            "samples",
            "bootstrap_resamples",
            "bootstrap_seed",
            "interval_counts",
            "batch_sizes",
            "shapes",
            "localities",
            "timed_exact_layer",
            "timed_scalar_layer",
            "excluded",
            "pair_order",
            "claim_scope",
        },
        "methodology",
    )
    samples = methodology["samples"]
    if type(samples) is not int or samples < MINIMUM_SAMPLES:
        raise ValueError("diagnostic sample count is invalid")
    fixed_method = {
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "timed_exact_layer": "one mutate_packed call through packed-result construction",
        "timed_scalar_layer": "ordered cpp_boundary RangeSet add/discard replay",
        "excluded": "manager/domain setup, operation packing, scalar oracle construction, result materialization, snapshots, validation, and artifact writing",
        "pair_order": "alternated exact-first and scalar-first by sample index",
        "claim_scope": "bounded diagnostic cells only; no universal threshold or stable gate",
    }
    if any(
        not _json_exact_equal(methodology[key], value)
        for key, value in fixed_method.items()
    ):
        raise ValueError("diagnostic fixed methodology does not match")
    for key in ("interval_counts", "batch_sizes", "shapes", "localities"):
        if type(methodology[key]) is not list:
            raise ValueError("diagnostic methodology matrix type is invalid")
    _expected_profile_dimensions(methodology)
    expected_cases = _cases(
        tuple(methodology["interval_counts"]),
        tuple(methodology["batch_sizes"]),
        tuple(methodology["shapes"]),
        tuple(methodology["localities"]),
    )

    manifest = _require_exact_keys(
        report["workload_manifest"],
        {"schema", "coordinate_semantics", "cases", "digest"},
        "workload manifest",
    )
    manifest_body = {key: value for key, value in manifest.items() if key != "digest"}
    if manifest["digest"] != _checksum(manifest_body):
        raise ValueError("workload digest does not match manifest")
    if not _json_exact_equal(manifest, workload_manifest(expected_cases)):
        raise ValueError("workload manifest does not match reconstructed matrix")

    rows = report["rows"]
    if type(rows) is not list or len(rows) != len(expected_cases):
        raise ValueError("diagnostic rows do not match manifest")
    row_keys = {
        "case_id",
        "interval_count",
        "batch_size",
        "shape",
        "locality",
        "exact_latency_ns_samples",
        "exact_latency_ns_median",
        "exact_latency_ns_confidence_95",
        "scalar_latency_ns_samples",
        "scalar_latency_ns_median",
        "scalar_latency_ns_confidence_95",
        "paired_exact_over_scalar_samples",
        "paired_exact_over_scalar_median",
        "paired_exact_over_scalar_confidence_95",
        "packed_input_bytes",
        "packed_result_bytes",
        "work",
        "validated_sample_count",
    }
    for row, case in zip(rows, expected_cases, strict=True):
        _require_exact_keys(row, row_keys, "row")
        expected_manifest = _case_manifest(case)
        reconstructed = {
            "case_id": expected_manifest["case_id"],
            "interval_count": expected_manifest["interval_count"],
            "batch_size": expected_manifest["batch_size"],
            "shape": expected_manifest["shape"],
            "locality": expected_manifest["locality"],
            "packed_input_bytes": expected_manifest["packed_input_bytes"],
            "packed_result_bytes": _expected_packed_result_bytes(case),
            "work": expected_manifest["work"],
            "validated_sample_count": samples,
        }
        if any(
            not _json_exact_equal(row[key], value)
            for key, value in reconstructed.items()
        ):
            raise ValueError("diagnostic row does not match reconstructed workload")
        exact = row["exact_latency_ns_samples"]
        scalar = row["scalar_latency_ns_samples"]
        if (
            type(exact) is not list
            or type(scalar) is not list
            or len(exact) != samples
            or len(scalar) != samples
            or any(type(value) is not int or value <= 0 for value in (*exact, *scalar))
        ):
            raise ValueError("diagnostic raw samples are invalid")
        ratios = [left / right for left, right in zip(exact, scalar, strict=True)]
        derived = {
            "exact_latency_ns_median": statistics.median(exact),
            "exact_latency_ns_confidence_95": _median_confidence(
                [float(value) for value in exact]
            ),
            "scalar_latency_ns_median": statistics.median(scalar),
            "scalar_latency_ns_confidence_95": _median_confidence(
                [float(value) for value in scalar]
            ),
            "paired_exact_over_scalar_samples": ratios,
            "paired_exact_over_scalar_median": statistics.median(ratios),
            "paired_exact_over_scalar_confidence_95": _median_confidence(ratios),
        }
        for key, value in derived.items():
            if not _json_exact_equal(row[key], value):
                raise ValueError(
                    f"diagnostic derived field {key} does not match raw samples"
                )
    _require_exact_keys(
        report["summary"],
        {
            "observed_median_break_even_batch",
            "cells_exact_faster_with_95_ci",
            "cells_scalar_faster_with_95_ci",
            "cells_inconclusive",
        },
        "summary",
    )
    if not _json_exact_equal(report["summary"], _summary(rows)):
        raise ValueError("diagnostic summary does not match rows")
    if markdown.read_text() != render_markdown(report, digest):
        raise ValueError("diagnostic Markdown does not match JSON")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "local"), default="smoke")
    parser.add_argument("--samples", type=int, default=MINIMUM_SAMPLES)
    parser.add_argument("--include-100000", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/experiments/exact-batch-application-matrix.json"),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify an existing triplet without running benchmark cells",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verify:
        report = verify_artifacts(args.output)
        print(
            f"verified cells={len(report['rows'])} "
            f"workload={report['workload_manifest']['digest']} output={args.output}"
        )
        return 0
    report = run_benchmark(
        profile=args.profile,
        samples=args.samples,
        include_100000=args.include_100000,
    )
    paths = write_artifacts(report, args.output)
    verify_artifacts(args.output)
    print(
        f"cells={len(report['rows'])} samples={args.samples} "
        f"workload={report['workload_manifest']['digest']} output={paths[0]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
