#!/usr/bin/env python3
"""Paired benchmark for the experimental exact whole-batch CPU mutation path."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from array import array
from pathlib import Path
from typing import Any

from tests.performance.mutation_attribution import paired_statistics
from treemendous import BackendUnavailableError, Span, create_range_set
from treemendous.experimental.exact_batch import ExactBatchRangeSet, MutationOpcode

BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
DOMAIN = (0, 1_024)
BASELINE_BACKEND = "cpp_boundary"
INITIAL_INTERVAL_COUNT = 64
ABSOLUTE_BATCH16_OPS_PER_SECOND = 2_000_000.0
BATCH16_SPEEDUP = 2.0
SCALAR_REGRESSION_LIMIT = 1.03

Operation = tuple[int, Span]
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


def _packed(trace: Trace) -> array[int]:
    return array(
        "q", (value for opcode, span in trace for value in (opcode, span.start, span.end))
    )


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
                require_covered=opcode
                == MutationOpcode.DISCARD_REQUIRE_COVERED,
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
                    require_covered=opcode
                    == MutationOpcode.DISCARD_REQUIRE_COVERED,
                )
            )
    return tuple(results)


def _assert_initial_state(snapshot: Any, *, label: str) -> None:
    if len(snapshot.intervals) != INITIAL_INTERVAL_COUNT:
        raise AssertionError(
            f"{label} contracted to {len(snapshot.intervals)} intervals; "
            f"expected {INITIAL_INTERVAL_COUNT}"
        )


def _validate(trace: Trace, packed_trace: array[int]) -> Any:
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


def _median_confidence(values: list[float], seed: int = 16) -> tuple[float, float]:
    rng = random.Random(seed)
    boot = sorted(
        statistics.median(rng.choices(values, k=len(values))) for _ in range(10_000)
    )
    return boot[249], boot[9_749]


def run_benchmark(*, samples: int = 30, target_operations: int = 20_000) -> dict[str, Any]:
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
        iterations = max(20, target_operations // batch_size)
        batch_samples: list[int] = []
        scalar_samples: list[int] = []
        for sample in range(samples):
            exact = _new_exact()
            scalar = _new_scalar()

            def time_exact() -> int:
                started = time.perf_counter_ns()
                for _ in range(iterations):
                    result = exact.mutate_packed(packed_trace)
                    del result
                return time.perf_counter_ns() - started

            def time_scalar() -> int:
                started = time.perf_counter_ns()
                for _ in range(iterations):
                    _scalar_once(scalar, trace)
                return time.perf_counter_ns() - started

            if sample % 2:
                scalar_elapsed = time_scalar()
                batch_elapsed = time_exact()
            else:
                batch_elapsed = time_exact()
                scalar_elapsed = time_scalar()
            if exact.snapshot() != expected_state or scalar.snapshot() != expected_state:
                raise AssertionError("repeated timed trace changed the 64-interval state")
            logical_operations = iterations * batch_size
            batch_samples.append(batch_elapsed // logical_operations)
            scalar_samples.append(scalar_elapsed // logical_operations)

        comparison = paired_statistics(
            scalar_samples, batch_samples, ratio_limit=1.0
        )
        throughput_samples = [1_000_000_000 / value for value in batch_samples]
        throughput_ci = _median_confidence(throughput_samples)
        rows[str(batch_size)] = {
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

    batch16 = rows["16"]
    gates = {
        "batch16_absolute": {
            "threshold_ops_per_second_lower_95": ABSOLUTE_BATCH16_OPS_PER_SECOND,
            "observed": batch16["batch_ops_per_second_confidence_95"][0],
            "passed": batch16["batch_ops_per_second_confidence_95"][0]
            >= ABSOLUTE_BATCH16_OPS_PER_SECOND,
        },
        "batch16_speedup": {
            "threshold_lower_95": BATCH16_SPEEDUP,
            "observed": batch16["speedup_confidence_95"][0],
            "passed": batch16["speedup_confidence_95"][0] >= BATCH16_SPEEDUP,
        },
        "break_even_by_4": {
            "threshold_lower_95": 1.0,
            "observed": rows["4"]["speedup_confidence_95"][0],
            "passed": rows["4"]["speedup_confidence_95"][0] >= 1.0,
        },
        "stable_scalar_regression": {
            "threshold_candidate_over_baseline": SCALAR_REGRESSION_LIMIT,
            "passed": None,
            "evidence": "Run mutation_attribution.py against clean baseline and candidate roots with --regression-ratio-limit 1.03.",
        },
    }
    return {
        "schema": "treemendous-experimental-exact-batch-v1",
        "baseline_backend": BASELINE_BACKEND,
        "methodology": {
            "samples": samples,
            "paired_order": "alternating within each paired sample",
            "initial_and_final_interval_count": INITIAL_INTERVAL_COUNT,
            "batch_sizes": BATCH_SIZES,
            "workload": "deterministic per-size restorative traces; four-operation blocks preserve the exact initial state",
            "batch_1": "separately labelled no-op call-overhead diagnostic",
            "timed_batch_layer": "buffer acquisition/copy, state staging, ordered execution, packed allocation, atomic commit, packed-result destruction",
            "excluded": "manager/setup construction, invariant snapshots, validation, and materialize",
        },
        "rows": rows,
        "materialize_16_ns_per_operation_samples": materialize_samples,
        "gates": gates,
    }


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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    callable_gates = [
        gate for gate in report["gates"].values() if gate["passed"] is not None
    ]
    return int(
        args.enforce_hard_gates
        and not all(bool(gate["passed"]) for gate in callable_gates)
    )


if __name__ == "__main__":
    raise SystemExit(main())
