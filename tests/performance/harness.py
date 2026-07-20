"""Correctness-checked benchmark harness for canonical range sets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from tests.performance.oracle import RangeOracle
from treemendous import Span, create_range_set


@dataclass(frozen=True)
class Operation:
    """One immutable operation in an ordered benchmark trace."""

    kind: str
    start: int | None = None
    end: int | None = None
    length: int | None = None
    not_before: int | None = None
    not_after: int | None = None
    job_id: int | None = None
    job_class: str | None = None
    expected_error: str | None = None


@dataclass(frozen=True)
class BenchmarkWorkload:
    """Setup and timed phases plus their explicit managed domain."""

    name: str
    domain: tuple[tuple[int, int], ...]
    setup: tuple[Operation, ...]
    operations: tuple[Operation, ...]
    coordinate_extent: int
    dimensions: tuple[tuple[str, str], ...] = ()
    initially_available: bool = False


@dataclass(frozen=True)
class ReplaySummary:
    """Complete normalized result required before a sample is accepted."""

    requested_operations: int
    successful_operations: int
    no_op_operations: int
    error_operations: int
    actual_interval_count: int
    total_available: int
    touched_intervals: int
    touched_length: int
    normalized_state: tuple[tuple[int, int], ...]
    query_results: tuple[tuple[str, int | str | None, int | str | None], ...]
    state_checksum: str
    query_checksum: str
    scheduling_success: tuple[tuple[str, int, int], ...]
    scheduling_fairness: float | None


@dataclass(frozen=True)
class _ObservedMutation:
    changed_length: int
    touched_intervals: int


@dataclass(frozen=True)
class Sample:
    """One timing sample accepted by a separate fully validated replay."""

    implementation: str
    setup_ns: int
    execution_ns: int
    validation_ns: int
    operation_latency_ns: tuple[tuple[str, tuple[int, ...]], ...]
    summary: ReplaySummary


@dataclass(frozen=True)
class TimingStatistics:
    """Robust directional statistics across independent benchmark runs."""

    independent_runs: int
    median_ns: float
    median_absolute_deviation_ns: float
    confidence_95_ns: tuple[float, float]
    p10_ns: float
    p90_ns: float


@dataclass(frozen=True)
class DescriptiveTimingStatistics:
    """Descriptive statistics for dependent operation invocations."""

    operation_invocations: int
    median_ns: float
    median_absolute_deviation_ns: float
    p10_ns: float
    p90_ns: float


class RangeLike(Protocol):
    def add(self, span: Span): ...

    def discard(self, span: Span): ...

    def first_fit(
        self, length: int, *, not_before: int, not_after: int | None = None
    ): ...

    def allocate(
        self, length: int, *, not_before: int, not_after: int | None = None
    ): ...

    def intervals(self): ...

    def overlaps(self, span: Span): ...

    def snapshot(self): ...

    def stats(self): ...


_ERROR_TYPES = {
    "ValueError": ValueError,
    "TypeError": TypeError,
    "OverflowError": OverflowError,
}


def _checksum(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _jain_fairness(success: dict[str, list[int]]) -> float | None:
    rates = [wins / attempts for attempts, wins in success.values() if attempts]
    if not rates:
        return None
    denominator = len(rates) * sum(rate * rate for rate in rates)
    return (sum(rates) ** 2 / denominator) if denominator else 1.0


def _normalized_state(target: RangeLike | RangeOracle) -> tuple[tuple[int, int], ...]:
    if isinstance(target, RangeOracle):
        return target.intervals
    return tuple((item.start, item.end) for item in target.intervals())


def _total(target: RangeLike | RangeOracle) -> int:
    if isinstance(target, RangeOracle):
        return target.total
    return target.snapshot().total_free


def _snapshot_observation(
    target: RangeLike | RangeOracle,
) -> tuple[int, str]:
    if isinstance(target, RangeOracle):
        state = target.intervals
        total = target.total
    else:
        snapshot = target.snapshot()
        state = tuple((item.start, item.end) for item in snapshot.intervals)
        total = snapshot.total_free
    return total, _checksum(state)


def _stats_observation(
    target: RangeLike | RangeOracle,
) -> tuple[int | None, ...]:
    if isinstance(target, RangeOracle):
        state = target.intervals
        largest = max((end - start for start, end in state), default=0)
        bounds = target.domain_bounds
        return (
            target.total,
            target.domain_total - target.total,
            target.domain_total,
            len(state),
            largest,
            *bounds,
        )
    stats = target.stats()
    return (
        stats.total_free,
        stats.total_occupied,
        stats.total_space,
        stats.free_chunks,
        stats.largest_chunk,
        *stats.bounds,
    )


def _execute_trace(
    target: RangeLike | RangeOracle,
    operations: tuple[Operation, ...],
    *,
    record_latencies: bool,
) -> tuple[ReplaySummary, dict[str, list[int]]]:
    successes = no_ops = errors = touched_intervals = touched_length = 0
    queries: list[tuple[str, int | str | None, int | str | None]] = []
    allocations: dict[int, tuple[int, int]] = {}
    scheduling: dict[str, list[int]] = {}
    latencies: dict[str, list[int]] = {}

    for operation in operations:
        started = time.perf_counter_ns() if record_latencies else 0
        try:
            mutation: Any = None
            result: tuple[int, int] | None = None
            query_hit = False
            if operation.kind == "add":
                assert operation.start is not None and operation.end is not None
                if isinstance(target, RangeOracle):
                    mutation = target.add(operation.start, operation.end)
                else:
                    mutation = target.add(Span(operation.start, operation.end))
            elif operation.kind == "discard":
                assert operation.start is not None and operation.end is not None
                if isinstance(target, RangeOracle):
                    mutation = target.discard(operation.start, operation.end)
                else:
                    mutation = target.discard(Span(operation.start, operation.end))
            elif operation.kind == "overlaps":
                assert operation.start is not None and operation.end is not None
                if isinstance(target, RangeOracle):
                    query_hit = target.overlaps(operation.start, operation.end)
                else:
                    query_hit = bool(
                        target.overlaps(Span(operation.start, operation.end))
                    )
                queries.append(("overlaps", int(query_hit), None))
            elif operation.kind == "snapshot":
                total, state_checksum = _snapshot_observation(target)
                queries.append(("snapshot", total, state_checksum))
            elif operation.kind == "stats":
                signature = _stats_observation(target)
                queries.append(("stats", _checksum(signature), signature[3]))
            elif operation.kind in {"first_fit", "allocate"}:
                assert operation.length is not None and operation.not_before is not None
                method = getattr(target, operation.kind)
                raw = method(
                    operation.length,
                    not_before=operation.not_before,
                    not_after=operation.not_after,
                )
                if operation.kind == "allocate" and isinstance(target, RangeOracle):
                    raw, mutation = raw
                result = (
                    None
                    if raw is None
                    else ((raw.start, raw.end) if hasattr(raw, "start") else raw)
                )
                if (
                    operation.kind == "allocate"
                    and result is not None
                    and not isinstance(target, RangeOracle)
                ):
                    # Atomic allocation returns its allocated range rather than
                    # MutationResult. Its geometric change is exactly that one
                    # contiguous result.
                    mutation = _ObservedMutation(operation.length, 1)
                queries.append(
                    (operation.kind, None, None)
                    if result is None
                    else (operation.kind, result[0], result[1])
                )
                if operation.job_id is not None and result is not None:
                    allocations[operation.job_id] = result
                if operation.job_class is not None:
                    attempts, wins = scheduling.setdefault(operation.job_class, [0, 0])
                    scheduling[operation.job_class] = [
                        attempts + 1,
                        wins + int(result is not None),
                    ]
            elif operation.kind == "cancel":
                assert operation.job_id is not None
                allocated = allocations.pop(operation.job_id, None)
                if allocated is not None:
                    mutation = (
                        target.add(*allocated)
                        if isinstance(target, RangeOracle)
                        else target.add(Span(*allocated))
                    )
            else:
                raise ValueError(f"unknown operation kind: {operation.kind}")
        except (ValueError, TypeError, OverflowError) as exc:
            if operation.expected_error != type(exc).__name__:
                raise
            errors += 1
            if operation.kind in {"first_fit", "allocate"}:
                queries.append((operation.kind, "error", type(exc).__name__))
        else:
            if operation.expected_error is not None:
                raise AssertionError(
                    f"{operation.kind} did not raise {operation.expected_error}"
                )
            changed_length = 0 if mutation is None else mutation.changed_length
            touched = (
                0
                if mutation is None
                else (
                    mutation.touched_intervals
                    if hasattr(mutation, "touched_intervals")
                    else len(mutation.changed)
                )
            )
            touched_length += changed_length
            touched_intervals += touched
            changed_or_found = changed_length > 0 or result is not None or query_hit
            if changed_or_found:
                successes += 1
            else:
                no_ops += 1
        finally:
            if record_latencies:
                latencies.setdefault(operation.kind, []).append(
                    time.perf_counter_ns() - started
                )

    state = _normalized_state(target)
    query_results = tuple(queries)
    schedule_tuple = tuple(
        (name, values[0], values[1]) for name, values in sorted(scheduling.items())
    )
    summary = ReplaySummary(
        requested_operations=len(operations),
        successful_operations=successes,
        no_op_operations=no_ops,
        error_operations=errors,
        actual_interval_count=len(state),
        total_available=_total(target),
        touched_intervals=touched_intervals,
        touched_length=touched_length,
        normalized_state=state,
        query_results=query_results,
        state_checksum=_checksum(state),
        query_checksum=_checksum(query_results),
        scheduling_success=schedule_tuple,
        scheduling_fairness=_jain_fairness(scheduling),
    )
    return summary, latencies


def _replay_public_operations(
    target: RangeLike, operations: tuple[Operation, ...]
) -> None:
    """Replay only declared public calls, without accounting or snapshots."""
    allocations: dict[int, tuple[int, int]] = {}
    for operation in operations:
        try:
            if operation.kind == "add":
                assert operation.start is not None and operation.end is not None
                target.add(Span(operation.start, operation.end))
            elif operation.kind == "discard":
                assert operation.start is not None and operation.end is not None
                target.discard(Span(operation.start, operation.end))
            elif operation.kind == "overlaps":
                assert operation.start is not None and operation.end is not None
                target.overlaps(Span(operation.start, operation.end))
            elif operation.kind == "snapshot":
                target.snapshot()
            elif operation.kind == "stats":
                target.stats()
            elif operation.kind in {"first_fit", "allocate"}:
                assert operation.length is not None and operation.not_before is not None
                raw = getattr(target, operation.kind)(
                    operation.length,
                    not_before=operation.not_before,
                    not_after=operation.not_after,
                )
                if operation.kind == "allocate" and operation.job_id is not None:
                    if raw is not None:
                        allocations[operation.job_id] = (raw.start, raw.end)
            elif operation.kind == "cancel":
                assert operation.job_id is not None
                allocated = allocations.pop(operation.job_id, None)
                if allocated is not None:
                    target.add(Span(*allocated))
            else:
                raise ValueError(f"unknown operation kind: {operation.kind}")
        except (ValueError, TypeError, OverflowError) as exc:
            if operation.expected_error != type(exc).__name__:
                raise
        else:
            if operation.expected_error is not None:
                raise AssertionError(
                    f"{operation.kind} did not raise {operation.expected_error}"
                )


def oracle_summary(workload: BenchmarkWorkload) -> tuple[int, ReplaySummary]:
    """Replay setup and timed traces in the independent model."""
    oracle = RangeOracle(
        workload.domain, initially_available=workload.initially_available
    )
    _execute_trace(oracle, workload.setup, record_latencies=False)
    initial_count = len(oracle.intervals)
    summary, _ = _execute_trace(oracle, workload.operations, record_latencies=False)
    return initial_count, summary


def _new_backend(backend_id: str, workload: BenchmarkWorkload) -> RangeLike:
    return create_range_set(
        domain=workload.domain,
        backend=backend_id,
        initially_available=workload.initially_available,
    )


def validate_target(
    target: RangeLike, workload: BenchmarkWorkload, *, name: str = "target"
) -> tuple[ReplaySummary, dict[str, list[int]]]:
    """Apply both phases and reject a target that differs from the oracle."""
    initial_count, expected = oracle_summary(workload)
    _execute_trace(target, workload.setup, record_latencies=False)
    observed_initial = len(target.intervals())
    if observed_initial != initial_count:
        raise AssertionError(
            f"{name} setup interval count {observed_initial} != oracle {initial_count}"
        )
    observed, latencies = _execute_trace(
        target, workload.operations, record_latencies=True
    )
    if observed != expected:
        raise AssertionError(
            f"{name} benchmark validation failed\n"
            f"expected={expected!r}\nobserved={observed!r}"
        )
    return observed, latencies


def run_validated_sample(backend_id: str, workload: BenchmarkWorkload) -> Sample:
    """Time a public replay and validate an equivalent independent replay."""
    validation_started = time.perf_counter_ns()
    initial_count, expected = oracle_summary(workload)
    validation_target = _new_backend(backend_id, workload)
    _execute_trace(validation_target, workload.setup, record_latencies=False)
    observed_initial = len(validation_target.intervals())
    if observed_initial != initial_count:
        raise AssertionError(
            f"{backend_id} setup interval count {observed_initial} != oracle {initial_count}"
        )
    observed, latencies = _execute_trace(
        validation_target, workload.operations, record_latencies=True
    )
    if observed != expected:
        raise AssertionError(
            f"{backend_id} benchmark validation failed\n"
            f"expected={expected!r}\nobserved={observed!r}"
        )
    validation_ns = time.perf_counter_ns() - validation_started

    setup_started = time.perf_counter_ns()
    target = _new_backend(backend_id, workload)
    _replay_public_operations(target, workload.setup)
    setup_ns = time.perf_counter_ns() - setup_started
    timed_initial = len(target.intervals())
    if timed_initial != initial_count:
        raise AssertionError(
            f"{backend_id} timed setup interval count {timed_initial} "
            f"!= oracle {initial_count}"
        )

    execution_started = time.perf_counter_ns()
    _replay_public_operations(target, workload.operations)
    execution_ns = time.perf_counter_ns() - execution_started
    return Sample(
        backend_id,
        setup_ns,
        execution_ns,
        validation_ns,
        tuple((kind, tuple(values)) for kind, values in sorted(latencies.items())),
        observed,
    )


def _run_round(
    args: tuple[int, tuple[str, ...], BenchmarkWorkload],
) -> tuple[Sample, ...]:
    round_number, backend_ids, workload = args
    order = list(backend_ids)
    random.Random(0x5EED + round_number).shuffle(order)
    return tuple(run_validated_sample(backend_id, workload) for backend_id in order)


def _warm_sampling_worker(
    backend_ids: tuple[str, ...], workload: BenchmarkWorkload, warmups: int
) -> None:
    """Warm every implementation inside the process that will take samples."""
    for backend_id in backend_ids:
        for _ in range(warmups):
            run_validated_sample(backend_id, workload)


def _percentile(values: list[int] | list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower] * 1.0
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def timing_statistics(values: list[int], *, seed: int = 42) -> TimingStatistics:
    """Bootstrap a median only across independent benchmark runs."""
    if not values:
        raise ValueError("at least one independent timing run is required")
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    rng = random.Random(seed)
    bootstrap = sorted(
        statistics.median(rng.choices(values, k=len(values))) for _ in range(2_000)
    )
    return TimingStatistics(
        independent_runs=len(values),
        median_ns=median,
        median_absolute_deviation_ns=statistics.median(deviations),
        confidence_95_ns=(
            _percentile(bootstrap, 0.025),
            _percentile(bootstrap, 0.975),
        ),
        p10_ns=_percentile(values, 0.10),
        p90_ns=_percentile(values, 0.90),
    )


def descriptive_timing_statistics(values: list[int]) -> DescriptiveTimingStatistics:
    """Summarize dependent invocations without an IID confidence interval."""
    if not values:
        raise ValueError("at least one operation timing is required")
    median = statistics.median(values)
    return DescriptiveTimingStatistics(
        operation_invocations=len(values),
        median_ns=median,
        median_absolute_deviation_ns=statistics.median(
            abs(value - median) for value in values
        ),
        p10_ns=_percentile(values, 0.10),
        p90_ns=_percentile(values, 0.90),
    )


def compact_replay_summary(summary: ReplaySummary) -> dict[str, Any]:
    """Serialize validation evidence without embedding the full large-scale state."""
    return {
        "requested_operations": summary.requested_operations,
        "successful_operations": summary.successful_operations,
        "no_op_operations": summary.no_op_operations,
        "error_operations": summary.error_operations,
        "actual_interval_count": summary.actual_interval_count,
        "total_available": summary.total_available,
        "touched_intervals": summary.touched_intervals,
        "touched_length": summary.touched_length,
        "query_observations": len(summary.query_results),
        "state_checksum": summary.state_checksum,
        "query_checksum": summary.query_checksum,
        "scheduling_success": summary.scheduling_success,
        "scheduling_fairness": summary.scheduling_fairness,
    }


def benchmark_backends(
    backend_ids: tuple[str, ...],
    workload: BenchmarkWorkload,
    *,
    samples: int = 20,
    warmups: int = 2,
    processes: int = 2,
) -> dict[str, Any]:
    """Benchmark in randomized order using independent validated samples."""
    if samples < 20:
        raise ValueError("publishable benchmark runs require at least 20 samples")
    if warmups < 1:
        raise ValueError("at least one warmup is required")
    if processes < 1:
        raise ValueError("processes must be positive")

    initial_count, expected = oracle_summary(workload)
    rounds = [(index, backend_ids, workload) for index in range(samples)]
    if processes == 1:
        _warm_sampling_worker(backend_ids, workload, warmups)
        round_samples = [_run_round(item) for item in rounds]
    else:
        with ProcessPoolExecutor(
            max_workers=processes,
            initializer=_warm_sampling_worker,
            initargs=(backend_ids, workload, warmups),
        ) as executor:
            round_samples = list(executor.map(_run_round, rounds))

    by_backend: dict[str, list[Sample]] = {backend_id: [] for backend_id in backend_ids}
    for round_result in round_samples:
        for sample in round_result:
            by_backend[sample.implementation].append(sample)

    results: dict[str, Any] = {}
    for backend_id, backend_samples in by_backend.items():
        operation_invocations: dict[str, list[int]] = {}
        operation_run_medians: dict[str, list[int]] = {}
        for sample in backend_samples:
            for kind, values in sample.operation_latency_ns:
                operation_invocations.setdefault(kind, []).extend(values)
                operation_run_medians.setdefault(kind, []).append(
                    round(statistics.median(values))
                )
        results[backend_id] = {
            "execution": asdict(
                timing_statistics([sample.execution_ns for sample in backend_samples])
            ),
            "setup": asdict(
                timing_statistics([sample.setup_ns for sample in backend_samples])
            ),
            "validation_overhead": asdict(
                timing_statistics([sample.validation_ns for sample in backend_samples])
            ),
            "operation_latency": {
                kind: {
                    "per_run_median": asdict(timing_statistics(run_medians)),
                    "invocation_distribution_descriptive": asdict(
                        descriptive_timing_statistics(operation_invocations[kind])
                    ),
                }
                for kind, run_medians in sorted(operation_run_medians.items())
            },
            "validation": compact_replay_summary(expected),
        }

    return {
        "label": "local directional measurements; not a universal performance claim",
        "workload": workload.name,
        "dimensions": dict(workload.dimensions),
        "dataset": {
            "actual_interval_count": initial_count,
            "coordinate_extent": workload.coordinate_extent,
            "timed_operations": len(workload.operations),
        },
        "methodology": {
            "warmups": warmups,
            "independent_runs": samples,
            "worker_processes": processes,
            "implementation_order": "deterministically randomized per sample round",
            "confidence": "95% run-level percentile bootstrap interval for the median",
            "execution_timing": (
                "only the declared public operation trace; accounting, snapshots, "
                "normalization, JSON checksums, and divergence rejection run in a "
                "separate equivalent validation replay"
            ),
            "validation_overhead": "reported separately and excluded from execution",
            "operation_latency": (
                "collected during the validation replay; confidence intervals use "
                "per-run medians and invocation distributions are descriptive only"
            ),
        },
        "environment": environment_metadata(),
        "results": results,
    }


def qualify_backends(
    backend_ids: tuple[str, ...], workload: BenchmarkWorkload
) -> dict[str, Any]:
    """Run one fully checked load qualification per backend.

    This is a scale and correctness gate, not a statistically sampled speed
    comparison. Each reported timed replay is accepted only after a separate
    oracle replay rejects any state, query, accounting, or checkpoint drift.
    """
    if not backend_ids:
        raise ValueError("at least one backend is required")
    initial_count, expected = oracle_summary(workload)
    results: dict[str, Any] = {}
    for backend_id in backend_ids:
        sample = run_validated_sample(backend_id, workload)
        if sample.summary != expected:
            raise AssertionError(f"{backend_id} load qualification summary drifted")
        results[backend_id] = {
            "setup_ns": sample.setup_ns,
            "execution_ns": sample.execution_ns,
            "operations_per_second": (
                len(workload.operations) * 1_000_000_000 / sample.execution_ns
                if sample.execution_ns
                else None
            ),
            "validation_overhead_ns": sample.validation_ns,
            "operation_latency_descriptive": {
                kind: asdict(descriptive_timing_statistics(list(values)))
                for kind, values in sample.operation_latency_ns
            },
        }
    return {
        "label": (
            "single-run correctness and load qualification; timings are local "
            "observations, not comparative claims"
        ),
        "workload": workload.name,
        "dimensions": dict(workload.dimensions),
        "dataset": {
            "actual_interval_count": initial_count,
            "coordinate_extent": workload.coordinate_extent,
            "setup_operations": len(workload.setup),
            "timed_operations": len(workload.operations),
        },
        "methodology": {
            "independent_runs": 1,
            "purpose": "large-scale semantic qualification",
            "execution_timing": (
                "one declared public-operation trace after an equivalent replay "
                "was checked completely against the independent oracle"
            ),
            "performance_interpretation": (
                "do not rank backends or infer a regression threshold from one run"
            ),
        },
        "environment": environment_metadata(),
        "validation": compact_replay_summary(expected),
        "results": results,
    }


def environment_metadata() -> dict[str, str]:
    """Collect reproducibility metadata without failing outside a Git checkout."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        commit = "unknown"
    return {
        "commit": commit,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": str(os.cpu_count() or "unknown"),
    }
