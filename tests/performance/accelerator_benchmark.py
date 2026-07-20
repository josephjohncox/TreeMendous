"""Shared correctness gate for experimental CUDA and Metal measurements."""

from __future__ import annotations

import importlib
import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from tests.performance.harness import (
    BenchmarkWorkload,
    environment_metadata,
    timing_statistics,
)
from tests.performance.workload import immutable_query_workload
from treemendous.backends import CATALOG_BY_ID, Runtime


class HardwareUnavailableError(RuntimeError):
    """Hardware extension, toolchain, or runtime is unavailable."""


_ACCELERATOR_MODULES = {
    "gpu_boundary_summary": (
        "treemendous.cpp.gpu.boundary_summary_gpu",
        "GPUBoundarySummaryManager",
    ),
    "metal_boundary_summary": (
        "treemendous.cpp.metal.boundary_summary_metal",
        "MetalBoundarySummaryManager",
    ),
}


@dataclass(frozen=True)
class _AcceleratorSample:
    implementation: str
    setup_ns: int
    execution_ns: int


def _truthy_device_flag(value: object) -> bool:
    return (isinstance(value, bool) and value) or (
        isinstance(value, str) and value.lower() == "true"
    )


def _validate_runtime_flags(backend_id: str, module: ModuleType) -> dict[str, Any]:
    """Require explicit module and runtime/device evidence before collection."""
    if backend_id == "gpu_boundary_summary":
        if not _truthy_device_flag(getattr(module, "GPU_AVAILABLE", None)):
            raise HardwareUnavailableError(
                "CUDA module does not declare GPU_AVAILABLE=true"
            )
        if not _truthy_device_flag(getattr(module, "CUDA_RUNTIME_VALIDATED", None)):
            raise HardwareUnavailableError(
                "CUDA module does not declare CUDA_RUNTIME_VALIDATED=true"
            )
        info_function = getattr(module, "get_cuda_device_info", None)
        if info_function is None:
            raise HardwareUnavailableError(
                "CUDA module has no device-info runtime check"
            )
        try:
            info = dict(info_function())
        except Exception as exc:
            raise HardwareUnavailableError(
                f"CUDA runtime validation failed: {exc}"
            ) from exc
        if not isinstance(info.get("device_count"), int) or info["device_count"] < 1:
            raise HardwareUnavailableError("CUDA runtime reports no usable device")
        return info

    if not _truthy_device_flag(getattr(module, "METAL_AVAILABLE", None)):
        raise HardwareUnavailableError(
            "Metal module does not declare METAL_AVAILABLE=true"
        )
    info_function = getattr(module, "get_metal_device_info", None)
    if info_function is None:
        raise HardwareUnavailableError("Metal module has no device-info runtime check")
    try:
        info = dict(info_function())
    except Exception as exc:
        raise HardwareUnavailableError(
            f"Metal device validation failed: {exc}"
        ) from exc
    if not _truthy_device_flag(info.get("available")):
        raise HardwareUnavailableError(str(info.get("error", "no usable Metal device")))
    if not info.get("device_name"):
        raise HardwareUnavailableError("Metal runtime did not identify a device")
    return info


def _summary_signature(summary: object) -> tuple[int, ...]:
    fields = (
        "total_free_length",
        "total_occupied_length",
        "interval_count",
        "largest_interval_length",
        "largest_interval_start",
        "smallest_interval_length",
        "total_gaps",
        "earliest_start",
        "latest_end",
    )
    try:
        return tuple(int(getattr(summary, field)) for field in fields)
    except (AttributeError, TypeError, ValueError) as exc:
        raise AssertionError(
            f"invalid accelerator summary result: {summary!r}"
        ) from exc


def _expected_summary(intervals: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    lengths = [end - start for start, end in intervals]
    total = sum(lengths)
    largest = max(lengths)
    managed_start = intervals[0][0]
    managed_end = intervals[-1][1]
    return (
        total,
        managed_end - managed_start - total,
        len(intervals),
        largest,
        next(
            start
            for (start, _), length in zip(intervals, lengths, strict=True)
            if length == largest
        ),
        min(lengths),
        sum(
            left_end < right_start
            for (_, left_end), (right_start, _) in zip(
                intervals, intervals[1:], strict=False
            )
        ),
        managed_start,
        managed_end,
    )


def _normalize_fit(result: object, length: int) -> tuple[int, int] | None:
    if result is None:
        return None
    try:
        if isinstance(result, dict):
            if not result:
                return None
            start = int(result["start"])
        elif hasattr(result, "start"):
            start = int(result.start)
        else:
            start = int(result[0])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise AssertionError(f"invalid accelerator fit result: {result!r}") from exc
    return start, start + length


def _expected_best_fit(
    intervals: tuple[tuple[int, int], ...], length: int
) -> tuple[int, int] | None:
    candidates = [
        (end - start - length, start)
        for start, end in intervals
        if end - start >= length
    ]
    if not candidates:
        return None
    _, start = min(candidates)
    return start, start + length


def _setup_raw(implementation: object, workload: BenchmarkWorkload) -> None:
    for operation in workload.setup:
        assert operation.start is not None and operation.end is not None
        implementation.release_interval(operation.start, operation.end)


def _snapshot(workload: BenchmarkWorkload) -> tuple[tuple[int, int], ...]:
    return tuple(
        (operation.start, operation.end)
        for operation in workload.setup
        if operation.start is not None and operation.end is not None
    )


def _load_accelerator_class(backend_id: str) -> tuple[type[Any], dict[str, Any]]:
    if backend_id not in _ACCELERATOR_MODULES:
        raise ValueError(f"unsupported accelerator benchmark backend: {backend_id}")
    if CATALOG_BY_ID[backend_id].runtime not in {Runtime.CUDA, Runtime.METAL}:
        raise ValueError(f"{backend_id} is not an accelerator backend")
    module_name, class_name = _ACCELERATOR_MODULES[backend_id]
    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError) as exc:
        raise HardwareUnavailableError(str(exc)) from exc
    device_info = _validate_runtime_flags(backend_id, module)
    implementation_class = getattr(module, class_name, None)
    if implementation_class is None:
        raise HardwareUnavailableError(f"{module_name} does not expose {class_name}")

    # Construction plus a forced synchronized kernel call validates resources,
    # command queues/contexts, upload, execution, synchronization, and readback.
    try:
        probe = implementation_class()
        probe.release_interval(0, 2)
        observed = _summary_signature(probe.compute_summary_gpu())
    except Exception as exc:
        raise HardwareUnavailableError(
            f"{backend_id} device/resource initialization failed: {exc}"
        ) from exc
    expected = (2, 0, 1, 2, 0, 2, 0, 0, 2)
    if observed != expected:
        raise HardwareUnavailableError(
            f"{backend_id} runtime validation returned {observed!r}, expected {expected!r}"
        )
    return implementation_class, device_info


def _cpu_factory() -> object:
    spec = CATALOG_BY_ID["cpp_boundary_summary"]
    return spec.loader()(**dict(spec.constructor_args))


def _sample_summary(
    name: str,
    factory: Callable[[], object],
    workload: BenchmarkWorkload,
    *,
    device: bool,
    repetitions: int,
) -> _AcceleratorSample:
    expected = _expected_summary(_snapshot(workload))
    setup_started = time.perf_counter_ns()
    implementation = factory()
    _setup_raw(implementation, workload)
    setup_ns = time.perf_counter_ns() - setup_started
    operation = (
        implementation.compute_summary_gpu if device else implementation.get_summary
    )
    started = time.perf_counter_ns()
    observed = tuple(_summary_signature(operation()) for _ in range(repetitions))
    execution_ns = time.perf_counter_ns() - started
    if observed != (expected,) * repetitions:
        raise AssertionError(
            f"{name} synchronized summary results differ from CPU/oracle: "
            f"observed={observed!r}, expected={expected!r}"
        )
    return _AcceleratorSample(name, setup_ns, execution_ns)


def _sample_best_fit(
    name: str,
    factory: Callable[[], object],
    workload: BenchmarkWorkload,
    *,
    device: bool,
) -> _AcceleratorSample:
    intervals = _snapshot(workload)
    lengths = tuple(
        operation.length
        for operation in workload.operations
        if operation.length is not None
    )
    expected = tuple(_expected_best_fit(intervals, length) for length in lengths)
    setup_started = time.perf_counter_ns()
    implementation = factory()
    _setup_raw(implementation, workload)
    setup_ns = time.perf_counter_ns() - setup_started
    operation = (
        implementation.find_best_fit_gpu if device else implementation.find_best_fit
    )
    started = time.perf_counter_ns()
    observed = tuple(
        _normalize_fit(operation(length, True), length) for length in lengths
    )
    execution_ns = time.perf_counter_ns() - started
    if observed != expected:
        raise AssertionError(
            f"{name} synchronized best-fit results differ from CPU/oracle: "
            f"observed={observed!r}, expected={expected!r}"
        )
    return _AcceleratorSample(name, setup_ns, execution_ns)


def benchmark_accelerator(
    backend_id: str,
    *,
    samples: int = 20,
    warmups: int = 2,
    intervals: int = 64,
    operations: int = 500,
) -> dict[str, Any]:
    """Measure forced synchronized device summary and best-fit operations."""
    if samples < 20:
        raise ValueError("accelerator benchmark runs require at least 20 samples")
    if warmups < 1:
        raise ValueError("at least one warmup is required")
    implementation_class, device_info = _load_accelerator_class(backend_id)
    workload = immutable_query_workload(
        interval_count=intervals, queries_per_snapshot=operations
    )
    factories: dict[str, tuple[Callable[[], object], bool]] = {
        "cpp_boundary_summary_cpu": (_cpu_factory, False),
        backend_id: (implementation_class, True),
    }
    operation_runners = {
        "synchronized_summary": lambda name, factory, device: _sample_summary(
            name,
            factory,
            workload,
            device=device,
            repetitions=operations,
        ),
        "synchronized_best_fit": lambda name, factory, device: _sample_best_fit(
            name, factory, workload, device=device
        ),
    }
    reports = []
    for operation_name, runner in operation_runners.items():
        for name, (factory, device) in factories.items():
            for _ in range(warmups):
                runner(name, factory, device)
        collected: dict[str, list[_AcceleratorSample]] = {
            name: [] for name in factories
        }
        for index in range(samples):
            order = list(factories)
            random.Random(0xACCE1 + index).shuffle(order)
            for name in order:
                factory, device = factories[name]
                collected[name].append(runner(name, factory, device))
        reports.append(
            {
                "label": (
                    "experimental synchronized device-operation timings; "
                    "local directional data only"
                ),
                "workload": workload.name,
                "operation": operation_name,
                "dataset": {
                    "actual_interval_count": len(_snapshot(workload)),
                    "coordinate_extent": workload.coordinate_extent,
                    "timed_device_invocations": operations,
                },
                "validation": (
                    "every ordered result validated against independent geometry "
                    "and the CPU runner"
                ),
                "implementations": {
                    name: {
                        "execution": asdict(
                            timing_statistics(
                                [sample.execution_ns for sample in values]
                            )
                        ),
                        "setup": asdict(
                            timing_statistics([sample.setup_ns for sample in values])
                        ),
                    }
                    for name, values in collected.items()
                },
            }
        )
    return {
        "schema": "treemendous-experimental-accelerator-benchmark-v2",
        "experimental_backend": backend_id,
        "experimental": True,
        "device": device_info,
        "environment": environment_metadata(),
        "methodology": {
            "warmups": warmups,
            "independent_runs": samples,
            "order": "randomized per independent run",
            "semantics": "CPU and device runners execute identical ordered calls",
            "device_timing": (
                "forced compute_summary_gpu/find_best_fit_gpu calls include API-level "
                "device synchronization and readback"
            ),
            "confidence": "95% run-level percentile bootstrap interval for the median",
        },
        "reports": reports,
    }


def write_and_print(report: dict[str, Any], output: Path | None) -> None:
    """Emit validation and raw confidence information without derived claims."""
    for workload in report["reports"]:
        print(f"\n{workload['operation']}: {workload['label']}")
        print(
            f"  actual intervals={workload['dataset']['actual_interval_count']}, "
            f"extent={workload['dataset']['coordinate_extent']}"
        )
        for name, result in workload["implementations"].items():
            timing = result["execution"]
            low, high = timing["confidence_95_ns"]
            print(
                f"  {name}: median={timing['median_ns'] / 1e6:.3f} ms, "
                f"95% run-median CI={low / 1e6:.3f}..{high / 1e6:.3f} ms"
            )
    if output is not None:
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
