"""Correctness tests for the benchmark oracle and trace semantics."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest

from treemendous import IntervalResult, Span, create_range_set
from tests.performance.accelerator_benchmark import (
    HardwareUnavailableError,
    _load_accelerator_class,
    _validate_runtime_flags,
    benchmark_accelerator,
)
from tests.performance.batch_operations_benchmark import (
    _validate_implementation,
    batch_workload,
    benchmark_batches,
    contiguous_same_operation_runs,
)
from tests.performance.harness import (
    BenchmarkWorkload,
    Operation,
    benchmark_backends,
    oracle_summary,
    run_validated_sample,
    timing_statistics,
    validate_target,
)
from tests.performance.workload import (
    fragmented_workload,
    scheduling_workload,
)

pytestmark = pytest.mark.unit


class _DivergentQueryTarget:
    def __init__(self, target):
        self._target = target

    def __getattr__(self, name):
        return getattr(self._target, name)

    def first_fit(self, length, *, not_before, not_after=None):
        found = self._target.first_fit(
            length, not_before=not_before, not_after=not_after
        )
        if found is None:
            return None
        return IntervalResult(found.start + 1, found.end + 1)


class _RawBatchImplementation:
    def __init__(self):
        self._target = create_range_set(
            ((0, 1_000_000),), backend="py_boundary", initially_available=False
        )

    def release_interval(self, start, end):
        self._target.add(Span(start, end))

    def reserve_interval(self, start, end):
        self._target.discard(Span(start, end))

    def batch_release(self, spans):
        for start, end in spans:
            self.release_interval(start, end)

    def batch_reserve(self, spans):
        for start, end in spans:
            self.reserve_interval(start, end)

    def get_intervals(self):
        return [(item.start, item.end) for item in self._target.intervals()]

    def get_total_available_length(self):
        return self._target.snapshot().total_free


class _DivergentBatchImplementation(_RawBatchImplementation):
    def batch_release(self, spans):
        return None

    def batch_reserve(self, spans):
        return None


def test_oracle_and_canonical_backend_validate_complete_trace():
    workload = fragmented_workload(interval_count=8, operation_count=60)
    sample = run_validated_sample("py_boundary", workload)
    initial_count, expected = oracle_summary(workload)

    assert initial_count == 8
    assert sample.summary == expected
    assert expected.requested_operations == 60
    assert expected.error_operations == 1
    assert (
        expected.successful_operations
        + expected.no_op_operations
        + expected.error_operations
        == expected.requested_operations
    )
    assert len(expected.state_checksum) == 64
    assert len(expected.query_checksum) == 64


def test_divergent_query_is_rejected_before_timing_can_be_reported():
    workload = fragmented_workload(interval_count=8, operation_count=60)
    target = create_range_set(
        workload.domain,
        backend="py_boundary",
        initially_available=False,
    )

    with pytest.raises(AssertionError, match="benchmark validation failed"):
        validate_target(_DivergentQueryTarget(target), workload, name="divergent")


@pytest.mark.parametrize("cores", [1, 8, 64])
@pytest.mark.parametrize("occupancy", [0.25, 0.50, 0.75])
def test_scheduling_trace_reports_success_and_fairness(cores, occupancy):
    workload = scheduling_workload(cores=cores, occupancy=occupancy, jobs=80)
    sample = run_validated_sample("py_boundary", workload)

    assert dict(workload.dimensions)["cores"] == str(cores)
    assert dict(workload.dimensions)["occupancy"] == f"{occupancy:.2f}"
    assert {name for name, _, _ in sample.summary.scheduling_success} <= {
        "short",
        "medium",
        "long",
    }
    assert sample.summary.scheduling_fairness is not None
    assert 0.0 <= sample.summary.scheduling_fairness <= 1.0
    latency_kinds = dict(sample.operation_latency_ns)
    assert "allocate" in latency_kinds
    assert "cancel" in latency_kinds


def test_statistics_use_robust_summary_and_confidence_interval():
    result = timing_statistics([10, 11, 12, 13, 1_000])

    assert result.independent_runs == 5
    assert result.median_ns == 12
    assert result.median_absolute_deviation_ns == 1
    assert result.confidence_95_ns[0] <= result.median_ns
    assert result.confidence_95_ns[1] >= result.median_ns


def test_batch_validation_observes_complete_backend_accounting():
    workload = batch_workload(interval_count=4, operation_count=40)

    observed = _validate_implementation(
        _RawBatchImplementation, workload, name="observed"
    )
    _, expected = oracle_summary(workload)

    assert observed == expected
    assert observed.error_operations == 1
    assert observed.touched_intervals > 0
    assert observed.successful_operations + observed.no_op_operations + 1 == 40


def test_divergent_batch_implementation_is_rejected_at_run_boundary():
    workload = BenchmarkWorkload(
        "divergent-batch",
        ((0, 100),),
        (Operation("add", start=0, end=10),),
        (
            Operation("discard", start=1, end=2),
            Operation("add", start=20, end=22),
        ),
        100,
    )

    with pytest.raises(AssertionError, match="batch run 0"):
        _validate_implementation(
            _DivergentBatchImplementation, workload, name="divergent"
        )


def test_batch_grouping_preserves_noncommuting_interleaved_order():
    workload = batch_workload(interval_count=4, operation_count=40)
    runs = contiguous_same_operation_runs(workload.operations)

    assert tuple(operation for run in runs for operation in run) == workload.operations
    assert all(len({operation.kind for operation in run}) == 1 for run in runs)
    assert all(
        left[0].kind != right[0].kind
        for left, right in zip(runs, runs[1:], strict=False)
    )


def test_cuda_requires_explicit_availability_and_runtime_flags():
    module = ModuleType("fake_cuda")
    module.GPU_AVAILABLE = True
    module.CUDA_RUNTIME_VALIDATED = False
    module.get_cuda_device_info = lambda: {"device_count": 1}

    with pytest.raises(HardwareUnavailableError, match="CUDA_RUNTIME_VALIDATED"):
        _validate_runtime_flags("gpu_boundary_summary", module)


def test_metal_resource_initialization_failure_is_unavailable(monkeypatch):
    module = ModuleType("fake_metal")
    module.METAL_AVAILABLE = True
    module.get_metal_device_info = lambda: {
        "available": "true",
        "device_name": "fake",
    }

    class BrokenMetal:
        def __init__(self):
            raise RuntimeError("missing metallib")

    module.MetalBoundarySummaryManager = BrokenMetal
    monkeypatch.setattr(
        "tests.performance.accelerator_benchmark.importlib.import_module",
        lambda _: module,
    )

    with pytest.raises(HardwareUnavailableError, match="missing metallib"):
        _load_accelerator_class("metal_boundary_summary")


def test_accelerator_runtime_probe_forces_synchronized_summary(monkeypatch):
    calls = []
    module = ModuleType("fake_metal")
    module.METAL_AVAILABLE = True
    module.get_metal_device_info = lambda: {
        "available": "true",
        "device_name": "fake",
    }

    class FakeSummaryManager:
        def release_interval(self, start, end):
            self.interval = (start, end)

        def compute_summary_gpu(self):
            calls.append(self.interval)
            return SimpleNamespace(
                total_free_length=2,
                total_occupied_length=0,
                interval_count=1,
                largest_interval_length=2,
                largest_interval_start=0,
                smallest_interval_length=2,
                total_gaps=0,
                earliest_start=0,
                latest_end=2,
            )

    module.MetalBoundarySummaryManager = FakeSummaryManager
    monkeypatch.setattr(
        "tests.performance.accelerator_benchmark.importlib.import_module",
        lambda _: module,
    )

    implementation, info = _load_accelerator_class("metal_boundary_summary")

    assert implementation is FakeSummaryManager
    assert info["device_name"] == "fake"
    assert len(calls) == 1
    assert calls[0][0] == 0
    assert calls[0][1] == 2


def test_multiprocess_harness_initializes_warmups_in_sampling_worker(monkeypatch):
    workload = fragmented_workload(interval_count=2, operation_count=2)
    initialized = []

    class InlineExecutor:
        def __init__(self, *, max_workers, initializer, initargs):
            initialized.append((max_workers, initargs[2]))
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def map(self, function, values):
            return map(function, values)

    monkeypatch.setattr("tests.performance.harness.ProcessPoolExecutor", InlineExecutor)

    report = benchmark_backends(
        ("py_boundary",), workload, samples=20, warmups=3, processes=2
    )

    assert len(initialized) == 1
    assert initialized[0][0] == 2
    assert initialized[0][1] == 3
    assert report["methodology"]["independent_runs"] == 20
    operation = report["results"]["py_boundary"]["operation_latency"]["add"]
    assert operation["per_run_median"]["independent_runs"] == 20
    assert "confidence_95_ns" not in operation["invocation_distribution_descriptive"]


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (
            benchmark_accelerator,
            {
                "backend_id": "metal_boundary_summary",
                "samples": 20,
                "warmups": 0,
                "intervals": 2,
                "operations": 2,
            },
        ),
        (
            benchmark_batches,
            {
                "backend_id": "metal_boundary_summary",
                "samples": 20,
                "warmups": 0,
                "interval_count": 2,
                "operation_count": 2,
            },
        ),
    ],
)
def test_specialized_benchmarks_require_positive_warmups(function, kwargs):
    with pytest.raises(ValueError, match="at least one warmup"):
        function(**kwargs)


def test_publishable_harness_requires_twenty_samples():
    workload = fragmented_workload(interval_count=2, operation_count=2)

    with pytest.raises(ValueError, match="at least 20 samples"):
        benchmark_backends(
            ("py_boundary",),
            workload,
            samples=19,
            warmups=1,
            processes=1,
        )
