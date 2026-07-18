from __future__ import annotations

import importlib
import os

import pytest

from treemendous.backends import CATALOG_BY_ID, Unavailable, probe_backend
from treemendous.cpp.metal.mixed import MixedBoundarySummaryManager


def test_cuda_catalog_is_experimental_capability_empty_and_unavailable() -> None:
    spec = CATALOG_BY_ID["gpu_boundary_summary"]
    assert not spec.capabilities
    state = probe_backend(spec)
    assert isinstance(state, Unavailable)
    assert "experimental" in state.reason


def test_metal_catalog_is_experimental_32_bit_and_unavailable() -> None:
    spec = CATALOG_BY_ID["metal_boundary_summary"]
    assert spec.coordinate_bits == 32
    assert not spec.capabilities
    state = probe_backend(spec)
    assert isinstance(state, Unavailable)
    assert "experimental" in state.reason


def test_mixed_metal_invalid_batch_does_not_partially_mutate_replicas() -> None:
    class Recorder:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        def reserve_interval(self, *args) -> None:
            self.calls.append(("reserve", args))

        def batch_reserve(self, intervals) -> None:
            self.calls.append(("batch_reserve", tuple(intervals)))

    cpu = Recorder()
    metal = Recorder()
    allocated = Recorder()
    mixed = MixedBoundarySummaryManager.__new__(MixedBoundarySummaryManager)
    mixed.sync_cpu = True
    mixed.__dict__["cpu_manager"] = cpu
    mixed.__dict__["metal_manager"] = metal
    mixed.__dict__["allocated_manager"] = allocated
    with pytest.raises(ValueError):
        mixed.batch_reserve([(0, 10), (5, 5)])
    assert not cpu.calls
    assert not metal.calls
    assert not allocated.calls


@pytest.mark.cuda
def test_cuda_kernel_summary_matches_cpu() -> None:
    if os.environ.get("TREEMENDOUS_CUDA_HARDWARE_TEST") != "1":
        pytest.skip("requires explicit CUDA hardware lane")
    module = importlib.import_module("treemendous.cpp.gpu.boundary_summary_gpu")
    # Production discovery remains disabled until this explicit hardware gate is
    # promoted after passing on supported CUDA architectures.
    assert not module.GPU_AVAILABLE
    assert not module.CUDA_RUNTIME_VALIDATED
    info = module.get_cuda_device_info()
    assert info["device_count"] > 0

    manager = module.GPUBoundarySummaryManager()
    manager.batch_release([(0, 10), (20, 50), (60, 65)])
    manager.batch_reserve([(4, 7), (25, 30)])
    cpu = manager.compute_summary_cpu()
    gpu = manager.compute_summary_gpu()

    integral_fields = (
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
    for field in integral_fields:
        assert getattr(gpu, field) == getattr(cpu, field), field
    floating_fields = (
        "avg_interval_length",
        "avg_gap_size",
        "fragmentation_index",
        "utilization",
    )
    for field in floating_fields:
        assert getattr(gpu, field) == pytest.approx(getattr(cpu, field)), field
    assert manager.get_performance_stats().gpu_operations == 1


def _assert_metal_summary_parity(cpu, gpu) -> None:
    integral_fields = (
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
    floating_fields = (
        "avg_interval_length",
        "avg_gap_size",
        "fragmentation_index",
        "utilization",
    )
    for field in integral_fields:
        assert getattr(gpu, field) == getattr(cpu, field), field
    for field in floating_fields:
        assert getattr(gpu, field) == pytest.approx(getattr(cpu, field)), field


def _cpu_best_fit(
    intervals: list[tuple[int, int]], length: int
) -> tuple[int, int] | None:
    candidates = [item for item in intervals if item[1] - item[0] >= length]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[1] - item[0] - length, item[0]))


@pytest.mark.metal
def test_metal_gpu_summary_and_best_fit_match_cpu_boundary_cases() -> None:
    if os.environ.get("TREEMENDOUS_METAL_HARDWARE_TEST") != "1":
        pytest.skip("requires explicit macOS Metal hardware lane")
    module = importlib.import_module("treemendous.cpp.metal.boundary_summary_metal")
    cases: tuple[tuple[str, list[tuple[int, int]], tuple[int, ...]], ...] = (
        ("empty", [], (1,)),
        ("singleton", [(0, 7)], (1, 7, 8)),
        ("fragmented", [(0, 4), (10, 13), (20, 30), (40, 44)], (3, 4, 5, 11)),
        ("int32-lower", [(-(2**31), -(2**31) + 8)], (1, 8, 9)),
        ("int32-upper", [(2**31 - 10, 2**31 - 5), (2**31 - 4, 2**31 - 1)], (3, 4, 5)),
    )

    for name, intervals, lengths in cases:
        # Construction loads the metallib resource installed beside the extension.
        manager = module.MetalBoundarySummaryManager()
        if intervals:
            manager.batch_release(intervals)
        _assert_metal_summary_parity(
            manager.compute_summary_cpu(), manager.compute_summary_gpu()
        )
        assert manager.get_intervals() == intervals, name
        for length in lengths:
            assert manager.find_best_fit_gpu(length, True) == _cpu_best_fit(
                intervals, length
            ), f"{name}: best-fit length {length}"


@pytest.mark.metal
def test_metal_scalar_batch_validation_and_accounting() -> None:
    if os.environ.get("TREEMENDOUS_METAL_HARDWARE_TEST") != "1":
        pytest.skip("requires explicit macOS Metal hardware lane")
    module = importlib.import_module("treemendous.cpp.metal.boundary_summary_metal")
    manager = module.MetalBoundarySummaryManager()
    manager.release_interval(0, 100)
    manager.reserve_interval(10, 20)
    manager.batch_reserve([(30, 40)])
    expected = [(0, 10), (20, 30), (40, 100)]
    assert manager.get_intervals() == expected
    stats = manager.get_performance_stats()
    assert stats.total_operations == 3
    before = manager.get_intervals()
    before_operations = stats.total_operations
    with pytest.raises(ValueError):
        manager.batch_release([(50, 60), (70, 70)])
    assert manager.get_intervals() == before
    assert manager.get_performance_stats().total_operations == before_operations
