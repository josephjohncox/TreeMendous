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
            self.calls = []

        def reserve_interval(self, *args) -> None:
            self.calls.append(("reserve", args))

        def batch_reserve(self, intervals) -> None:
            self.calls.append(("batch_reserve", tuple(intervals)))

    mixed = MixedBoundarySummaryManager.__new__(MixedBoundarySummaryManager)
    mixed.sync_cpu = True
    mixed.cpu_manager = Recorder()
    mixed.metal_manager = Recorder()
    mixed.allocated_manager = Recorder()
    with pytest.raises(ValueError):
        mixed.batch_reserve([(0, 10), (5, 5)])
    assert not mixed.cpu_manager.calls
    assert not mixed.metal_manager.calls
    assert not mixed.allocated_manager.calls


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
