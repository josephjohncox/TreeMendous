#!/usr/bin/env python3
"""Semantics-preserving benchmark for experimental contiguous batch APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, cast

from tests.performance.accelerator_benchmark import (
    HardwareUnavailableError,
    _load_accelerator_class,
)
from tests.performance.harness import (
    BenchmarkWorkload,
    Operation,
    ReplaySummary,
    environment_metadata,
    oracle_summary,
    timing_statistics,
)
from treemendous.backends.normalize import normalize_intervals


class BatchImplementation(Protocol):
    def release_interval(self, start: int, end: int) -> None: ...
    def reserve_interval(self, start: int, end: int) -> None: ...
    def batch_release(self, intervals: list[tuple[int, int]]) -> None: ...
    def batch_reserve(self, intervals: list[tuple[int, int]]) -> None: ...
    def get_intervals(self) -> Any: ...
    def get_total_available_length(self) -> int: ...


def contiguous_same_operation_runs(
    operations: tuple[Operation, ...],
) -> tuple[tuple[Operation, ...], ...]:
    """Group only adjacent equal mutations; never move an operation."""
    runs: list[list[Operation]] = []
    for operation in operations:
        if operation.kind not in {"add", "discard"}:
            raise ValueError("batch traces may contain only add/discard mutations")
        if not runs or runs[-1][0].kind != operation.kind:
            runs.append([operation])
        else:
            runs[-1].append(operation)
    return tuple(tuple(run) for run in runs)


def batch_workload(
    *, interval_count: int = 64, operation_count: int = 500, seed: int = 42
) -> BenchmarkWorkload:
    """Build an interleaved trace with explicit contiguous batching boundaries."""
    if interval_count <= 0 or operation_count <= 0:
        raise ValueError("interval and operation counts must be positive")
    extent = interval_count * 8
    setup = tuple(
        Operation("add", start=index * 8, end=index * 8 + 4)
        for index in range(interval_count)
    )
    rng = random.Random(seed)
    operations: list[Operation] = []
    kind = "discard"
    while len(operations) < operation_count - 1:
        run_length = min(rng.randint(1, 8), operation_count - 1 - len(operations))
        for _ in range(run_length):
            length = rng.randint(1, 16)
            start = rng.randint(0, extent - length)
            operations.append(Operation(kind, start=start, end=start + length))
        kind = "add" if kind == "discard" else "discard"
    # A singleton invalid run proves that the implementation's error and atomic
    # no-mutation behavior are observed rather than copied from the oracle.
    error_kind = "add" if operations and operations[-1].kind == "discard" else "discard"
    operations.append(
        Operation(
            error_kind, start=extent // 2, end=extent // 2, expected_error="ValueError"
        )
    )
    return BenchmarkWorkload(
        "contiguous-transactional-bulk-mutations",
        ((0, extent),),
        setup,
        tuple(operations),
        extent,
        (("batching", "contiguous same-operation runs only"),),
    )


def _new_raw(backend_id: str) -> BatchImplementation:
    implementation_class, _ = _load_accelerator_class(backend_id)
    implementation = implementation_class()
    if not all(
        hasattr(implementation, method) for method in ("batch_release", "batch_reserve")
    ):
        raise HardwareUnavailableError(
            f"{backend_id} does not expose both transactional batch methods"
        )
    return cast(BatchImplementation, implementation)


def _setup(implementation: BatchImplementation, workload: BenchmarkWorkload) -> None:
    for operation in workload.setup:
        assert operation.start is not None and operation.end is not None
        implementation.release_interval(operation.start, operation.end)


def _state(implementation: BatchImplementation) -> tuple[tuple[int, int], ...]:
    return tuple(
        (interval.start, interval.end)
        for interval in normalize_intervals(implementation.get_intervals())
    )


def _total(implementation: BatchImplementation) -> int:
    try:
        return int(implementation.get_total_available_length())
    except (AttributeError, TypeError, ValueError) as exc:
        raise AssertionError("implementation returned an invalid total") from exc


def _checksum(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _changed_geometry(
    before: tuple[tuple[int, int], ...],
    start: int,
    end: int,
    *,
    adding: bool,
) -> tuple[tuple[int, int], ...]:
    if not adding:
        return tuple(
            (max(interval_start, start), min(interval_end, end))
            for interval_start, interval_end in before
            if interval_start < end and start < interval_end
        )
    cursor = start
    changed: list[tuple[int, int]] = []
    for interval_start, interval_end in before:
        if interval_end <= cursor:
            continue
        if interval_start >= end:
            break
        if interval_start > cursor:
            changed.append((cursor, min(interval_start, end)))
        cursor = max(cursor, interval_end)
        if cursor >= end:
            break
    if cursor < end:
        changed.append((cursor, end))
    return tuple(changed)


def _fully_covered(before: tuple[tuple[int, int], ...], start: int, end: int) -> bool:
    cursor = start
    for interval_start, interval_end in before:
        if interval_end <= cursor:
            continue
        if interval_start > cursor:
            return False
        cursor = max(cursor, interval_end)
        if cursor >= end:
            return True
    return False


def _changed_components(
    before: tuple[tuple[int, int], ...], start: int, end: int
) -> int:
    cursor = start
    components = 0
    for interval_start, interval_end in before:
        if interval_end <= cursor:
            continue
        if interval_start >= end:
            break
        if interval_start > cursor:
            components += 1
        cursor = max(cursor, interval_end)
        if cursor >= end:
            break
    return components + (1 if cursor < end else 0)


def _apply_scalar(implementation: BatchImplementation, operation: Operation) -> None:
    assert operation.start is not None and operation.end is not None
    if operation.kind == "add":
        implementation.release_interval(operation.start, operation.end)
    else:
        implementation.reserve_interval(operation.start, operation.end)


def _execute_scalar(
    implementation: BatchImplementation, operations: tuple[Operation, ...]
) -> None:
    for operation in operations:
        try:
            _apply_scalar(implementation, operation)
        except (ValueError, TypeError, OverflowError) as exc:
            if operation.expected_error != type(exc).__name__:
                raise
        else:
            if operation.expected_error is not None:
                raise AssertionError(
                    f"scalar {operation.kind} did not raise {operation.expected_error}"
                )


def _execute_batch_run(
    implementation: BatchImplementation, run: tuple[Operation, ...]
) -> None:
    spans: list[tuple[int, int]] = []
    for operation in run:
        assert operation.start is not None and operation.end is not None
        spans.append((operation.start, operation.end))
    expected_errors = [
        operation.expected_error for operation in run if operation.expected_error
    ]
    if expected_errors and len(run) != 1:
        raise AssertionError("expected-error batch operations must be singleton runs")
    try:
        if run[0].kind == "add":
            implementation.batch_release(spans)
        else:
            implementation.batch_reserve(spans)
    except (ValueError, TypeError, OverflowError) as exc:
        if expected_errors != [type(exc).__name__]:
            raise
    else:
        if expected_errors:
            raise AssertionError(
                f"batch {run[0].kind} did not raise {expected_errors[0]}"
            )


def _execute_batch(
    implementation: BatchImplementation, operations: tuple[Operation, ...]
) -> None:
    for run in contiguous_same_operation_runs(operations):
        _execute_batch_run(implementation, run)


def _observe_scalar(
    factory: Callable[[], BatchImplementation], workload: BenchmarkWorkload
) -> ReplaySummary:
    implementation = factory()
    _setup(implementation, workload)
    successes = no_ops = errors = touched_intervals = touched_length = 0
    queries: list[tuple[str, int | str | None, int | str | None]] = []
    for operation in workload.operations:
        assert operation.start is not None and operation.end is not None
        before = _state(implementation)
        before_total = _total(implementation)
        try:
            _apply_scalar(implementation, operation)
        except (ValueError, TypeError, OverflowError) as exc:
            if operation.expected_error != type(exc).__name__:
                raise
            errors += 1
            if _state(implementation) != before:
                raise AssertionError(
                    "failed scalar mutation changed implementation state"
                ) from exc
            if _total(implementation) != before_total:
                raise AssertionError(
                    "failed scalar mutation changed implementation total"
                ) from exc
            continue
        if operation.expected_error is not None:
            raise AssertionError(
                f"scalar {operation.kind} did not raise {operation.expected_error}"
            )
        after_total = _total(implementation)
        changed = abs(after_total - before_total)
        changed_geometry = _changed_geometry(
            before,
            operation.start,
            operation.end,
            adding=operation.kind == "add",
        )
        queries.append(
            (
                f"{operation.kind}:mutation",
                1 if _fully_covered(before, operation.start, operation.end) else 0,
                _checksum((changed_geometry, changed)),
            )
        )
        if operation.kind == "discard":
            touched = sum(
                interval_start < operation.end and interval_end > operation.start
                for interval_start, interval_end in before
            )
        else:
            touched = _changed_components(before, operation.start, operation.end)
        touched_intervals += touched
        touched_length += changed
        if changed:
            successes += 1
        else:
            no_ops += 1

    state = _state(implementation)
    query_results = tuple(queries)
    return ReplaySummary(
        requested_operations=len(workload.operations),
        successful_operations=successes,
        no_op_operations=no_ops,
        error_operations=errors,
        actual_interval_count=len(state),
        total_available=_total(implementation),
        touched_intervals=touched_intervals,
        touched_length=touched_length,
        normalized_state=state,
        query_results=query_results,
        state_checksum=_checksum(state),
        query_checksum=_checksum(query_results),
        scheduling_success=(),
        scheduling_fairness=None,
    )


def _validate_implementation(
    factory: Callable[[], BatchImplementation],
    workload: BenchmarkWorkload,
    *,
    name: str,
) -> ReplaySummary:
    initial_count, expected = oracle_summary(workload)
    observed = _observe_scalar(factory, workload)
    if observed != expected:
        raise AssertionError(
            f"{name} scalar accounting/intermediate semantics differ from oracle\n"
            f"expected={expected!r}\nobserved={observed!r}"
        )

    batched = factory()
    witness = factory()
    _setup(batched, workload)
    _setup(witness, workload)
    if len(_state(batched)) != initial_count or _state(batched) != _state(witness):
        raise AssertionError("batch setup differs from scalar implementation/oracle")
    for index, run in enumerate(contiguous_same_operation_runs(workload.operations)):
        _execute_scalar(witness, run)
        _execute_batch_run(batched, run)
        batch_observation = (_state(batched), _total(batched))
        scalar_observation = (_state(witness), _total(witness))
        if batch_observation != scalar_observation:
            raise AssertionError(
                f"{name} batch run {index} differs from ordered scalar intermediate "
                f"semantics: batch={batch_observation!r}, scalar={scalar_observation!r}"
            )
    if (_state(batched), _total(batched)) != (
        expected.normalized_state,
        expected.total_available,
    ):
        raise AssertionError(f"{name} batch final semantics differ from oracle")
    return observed


def _validated_timing(
    factory: Callable[[], BatchImplementation],
    workload: BenchmarkWorkload,
    *,
    name: str,
    batched: bool,
) -> tuple[int, int, ReplaySummary]:
    observed = _validate_implementation(factory, workload, name=name)
    setup_started = time.perf_counter_ns()
    implementation = factory()
    _setup(implementation, workload)
    setup_ns = time.perf_counter_ns() - setup_started
    started = time.perf_counter_ns()
    if batched:
        _execute_batch(implementation, workload.operations)
    else:
        _execute_scalar(implementation, workload.operations)
    execution_ns = time.perf_counter_ns() - started
    if (_state(implementation), _total(implementation)) != (
        observed.normalized_state,
        observed.total_available,
    ):
        raise AssertionError(
            "timed batch/scalar execution differs from validated state"
        )
    return setup_ns, execution_ns, observed


def benchmark_batches(
    backend_id: str,
    *,
    samples: int = 20,
    warmups: int = 2,
    interval_count: int = 64,
    operation_count: int = 500,
) -> dict[str, Any]:
    """Measure raw contiguous-batch API calls after complete semantic checks."""
    if samples < 20:
        raise ValueError("batch benchmark runs require at least 20 samples")
    if warmups < 1:
        raise ValueError("at least one warmup is required")
    workload = batch_workload(
        interval_count=interval_count, operation_count=operation_count
    )
    implementation_class, device_info = _load_accelerator_class(backend_id)
    factory = cast(Callable[[], BatchImplementation], implementation_class)
    probe = factory()
    if not all(hasattr(probe, method) for method in ("batch_release", "batch_reserve")):
        raise HardwareUnavailableError(
            f"{backend_id} does not expose both transactional batch methods"
        )
    observed = _validate_implementation(factory, workload, name=backend_id)
    for batched in (False, True):
        for _ in range(warmups):
            _validated_timing(factory, workload, name=backend_id, batched=batched)
    collected: dict[str, list[int]] = {"scalar": [], "contiguous_batch": []}
    setup_times: dict[str, list[int]] = {"scalar": [], "contiguous_batch": []}
    for index in range(samples):
        modes = ["scalar", "contiguous_batch"]
        random.Random(0xBA7C + index).shuffle(modes)
        for mode in modes:
            setup_ns, execution_ns, sample_observed = _validated_timing(
                factory,
                workload,
                name=backend_id,
                batched=mode == "contiguous_batch",
            )
            if sample_observed != observed:
                raise AssertionError("implementation accounting changed between runs")
            setup_times[mode].append(setup_ns)
            collected[mode].append(execution_ns)
    initial_count, _ = oracle_summary(workload)
    return {
        "schema": "treemendous-contiguous-batch-benchmark-v2",
        "label": (
            "local raw batch-API timings; host/device placement is not claimed; "
            "experimental backend"
        ),
        "backend": backend_id,
        "experimental": True,
        "device": device_info,
        "dataset": {
            "actual_interval_count": initial_count,
            "coordinate_extent": workload.coordinate_extent,
            "timed_operations": len(workload.operations),
            "contiguous_runs": len(contiguous_same_operation_runs(workload.operations)),
        },
        "validation": asdict(observed),
        "methodology": {
            "warmups": warmups,
            "independent_runs": samples,
            "order": "randomized scalar/batch order",
            "batch_semantics": "only contiguous same-operation runs are grouped",
            "accounting": (
                "success/no-op/error/touched values observed from scalar backend calls; "
                "batch state/total checked at every contiguous-run boundary"
            ),
            "confidence": "95% run-level percentile bootstrap interval for the median",
        },
        "environment": environment_metadata(),
        "results": {
            mode: {
                "execution": asdict(timing_statistics(values)),
                "setup": asdict(timing_statistics(setup_times[mode])),
            }
            for mode, values in collected.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="metal_boundary_summary",
        choices=("metal_boundary_summary", "gpu_boundary_summary"),
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--intervals", type=int, default=64)
    parser.add_argument("--operations", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = benchmark_batches(
            args.backend,
            samples=args.samples,
            warmups=args.warmups,
            interval_count=args.intervals,
            operation_count=args.operations,
        )
    except HardwareUnavailableError as exc:
        print(f"batch benchmark unavailable: {exc}")
        return 2
    print(report["label"])
    print(json.dumps(report["dataset"], indent=2))
    for mode, result in report["results"].items():
        timing = result["execution"]
        print(f"{mode}: median={timing['median_ns'] / 1e6:.3f} ms")
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
