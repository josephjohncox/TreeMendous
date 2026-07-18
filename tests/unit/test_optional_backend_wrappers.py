"""No-hardware contracts for optional-backend Python wrappers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from treemendous.basic.boundary_summary import (
    BoundarySummary,
    BoundarySummaryManager,
    demo_boundary_summary_performance,
)
from treemendous.cpp import gpu, metal
from treemendous.cpp.metal import mixed


def _expect_equal(actual: Any, expected: Any) -> None:
    if actual != expected:
        pytest.fail(f"expected {expected!r}, got {actual!r}")


class FakeMetalManager:
    """CPU-backed stand-in that exercises mixed-wrapper routing only."""

    def __init__(self) -> None:
        self.delegate = BoundarySummaryManager()
        self.info_printed = False

    @staticmethod
    def _tuple(result: Any) -> tuple[int, int] | None:
        if result is None:
            return None
        return result.start, result.end

    def release_interval(self, start: int, end: int) -> None:
        self.delegate.release_interval(start, end)

    def reserve_interval(self, start: int, end: int) -> None:
        self.delegate.reserve_interval(start, end)

    def batch_reserve(self, intervals) -> None:
        for start, end in intervals:
            self.reserve_interval(start, end)

    def batch_release(self, intervals) -> None:
        for start, end in intervals:
            self.release_interval(start, end)

    def find_interval(self, start: int, length: int) -> tuple[int, int] | None:
        return self._tuple(self.delegate.find_interval(start, length))

    def find_best_fit(
        self, length: int, prefer_early: bool = True
    ) -> tuple[int, int] | None:
        return self._tuple(self.delegate.find_best_fit(length, prefer_early))

    def find_largest_available(self) -> tuple[int, int] | None:
        return self._tuple(self.delegate.find_largest_available())

    def get_intervals(self) -> list[tuple[int, int]]:
        return [(item.start, item.end) for item in self.delegate.get_intervals()]

    def compute_summary_gpu(self) -> BoundarySummary:
        return self.delegate.get_summary()

    def compute_summary_cpu(self) -> BoundarySummary:
        return self.delegate.get_summary()

    def get_performance_stats(self) -> dict[str, bool]:
        return {"fake": True}

    def print_info(self) -> None:
        self.info_printed = True


def test_cuda_diagnostics_and_factory_are_truthful(monkeypatch) -> None:
    assert not gpu.is_gpu_available()
    monkeypatch.setattr(gpu, "boundary_summary_gpu", None)
    info = gpu.get_gpu_info()
    assert not info["available"]
    assert not info["extension_built"]
    assert "experimental" in info["error"]
    with pytest.raises(ImportError, match="experimental"):
        gpu.create_gpu_manager()

    extension = SimpleNamespace(get_cuda_device_info=lambda: {"name": "fake"})
    monkeypatch.setattr(gpu, "boundary_summary_gpu", extension)
    assert gpu.get_gpu_info()["device"] == {"name": "fake"}

    def broken_info() -> None:
        raise RuntimeError("device probe failed")

    extension.get_cuda_device_info = broken_info
    assert gpu.get_gpu_info()["device_error"] == "device probe failed"


def test_metal_diagnostics_factories_and_legacy_benchmark(monkeypatch) -> None:
    monkeypatch.setattr(metal, "IS_MACOS", False)
    info = metal.get_metal_info()
    assert not info["available"]
    _expect_equal(info["error"], "Metal is only available on macOS")

    monkeypatch.setattr(metal, "IS_MACOS", True)
    monkeypatch.setattr(metal, "METAL_AVAILABLE", False)
    monkeypatch.setattr(metal, "boundary_summary_metal", None)
    assert not metal.is_metal_available()
    assert "build-metal" in metal.get_metal_info()["error"]
    with pytest.raises(ImportError, match="build-metal"):
        metal.create_metal_manager()
    with pytest.raises(ImportError, match="build-metal"):
        metal.create_mixed_metal_manager()
    with pytest.warns(DeprecationWarning, match="experimental"):
        assert metal.benchmark_metal_speedup() == {
            "error": "Metal not available",
            "available": False,
        }

    class RawManager:
        pass

    extension = SimpleNamespace(
        MetalBoundarySummaryManager=RawManager,
        get_metal_device_info=lambda: {"available": True, "name": "fake"},
        benchmark_metal_speedup=lambda intervals, operations: {
            "intervals": intervals,
            "operations": operations,
            "speedup": 1.25,
        },
    )
    monkeypatch.setattr(metal, "boundary_summary_metal", extension)
    monkeypatch.setattr(metal, "METAL_AVAILABLE", True)
    assert metal.is_metal_available()
    assert metal.get_metal_info()["name"] == "fake"
    assert isinstance(metal.create_metal_manager(), RawManager)
    with pytest.warns(DeprecationWarning):
        assert metal.benchmark_metal_speedup(12, 7)["speedup"] == 1.25
    assert "benchmark_metal_speedup" in metal.__all__

    del extension.benchmark_metal_speedup
    with pytest.warns(DeprecationWarning):
        unavailable_benchmark = metal.benchmark_metal_speedup()
    assert unavailable_benchmark["available"]
    assert "does not provide" in unavailable_benchmark["error"]

    def broken_device_info() -> None:
        raise RuntimeError("probe failed")

    extension.get_metal_device_info = broken_device_info
    assert metal.get_metal_info() == {"available": False, "error": "probe failed"}


def test_metal_mixed_factory_forwards_configuration(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    class Factory:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(metal, "METAL_AVAILABLE", True)
    monkeypatch.setattr(metal, "MixedBoundarySummaryManager", Factory)
    created = metal.create_mixed_metal_manager(
        summary_path="gpu",
        best_fit_path="cpu",
        best_fit_min_intervals=3,
        summary_min_intervals=4,
        sync_cpu=True,
        track_allocations=False,
    )
    assert isinstance(created, Factory)
    assert calls == [
        {
            "summary_path": "gpu",
            "best_fit_path": "cpu",
            "best_fit_min_intervals": 3,
            "summary_min_intervals": 4,
            "sync_cpu": True,
            "track_allocations": False,
        }
    ]


def test_mixed_manager_cpu_paths_and_replication(monkeypatch, capsys) -> None:
    monkeypatch.setattr(mixed, "MetalBoundarySummaryManager", FakeMetalManager)
    manager = mixed.MixedBoundarySummaryManager(
        summary_path="cpu",
        best_fit_path="cpu",
        sync_cpu=True,
        track_allocations=True,
    )
    manager.release_interval(0, 100, "free")
    manager.reserve_interval(20, 30, "job")
    manager.batch_reserve(((40, 50),))
    manager.batch_release(((45, 50),))

    expected_intervals = [(0, 20), (30, 40), (45, 100)]
    actual_intervals = [(item.start, item.end) for item in manager.get_intervals()]
    _expect_equal(actual_intervals, expected_intervals)
    _expect_equal(manager.metal_manager.get_intervals(), expected_intervals)
    allocated = manager.get_allocated_intervals()
    _expect_equal(
        [(item.start, item.end, item.data) for item in allocated],
        [(20, 30, "job"), (40, 45, None)],
    )

    interval = manager.find_interval(0, 10)
    if interval is None:
        pytest.fail("expected a CPU-path interval")
    assert interval.start == 0
    assert manager.find_best_fit(5, prefer_early=True).start == 0
    assert manager.find_best_fit(5, prefer_early=False).start == 30
    assert manager.find_largest_available().start == 45
    assert manager.get_total_available_length() == 85
    assert manager.get_summary().total_free_length == 85
    assert manager.get_availability_stats()["total_free"] == 85
    assert manager.compute_summary_gpu().total_free_length == 85
    assert manager.compute_summary_cpu().total_free_length == 85
    assert manager.get_performance_stats() == {"fake": True}
    manager.print_info()
    assert manager.metal_manager.info_printed
    assert capsys.readouterr().out == ""


def test_mixed_manager_gpu_only_paths(monkeypatch) -> None:
    monkeypatch.setattr(mixed, "MetalBoundarySummaryManager", FakeMetalManager)
    manager = mixed.MixedBoundarySummaryManager(
        summary_path="cpu",
        best_fit_path="cpu",
        sync_cpu=False,
        track_allocations=False,
    )
    assert manager.summary_path == "gpu"
    assert manager.best_fit_path == "gpu"
    manager.release_interval(0, 30)
    manager.reserve_interval(10, 20)

    assert manager._interval_count() == 0
    assert manager._should_use_gpu_best_fit()
    assert manager._should_use_gpu_summary()
    interval = manager.find_interval(0, 5)
    if interval is None:
        pytest.fail("expected a GPU-path interval")
    assert interval.start == 0
    assert manager.find_best_fit(5, prefer_early=True).start == 0
    assert manager.find_best_fit(5, prefer_early=False).start == 0
    assert manager.find_largest_available().start == 0
    _expect_equal(manager.get_intervals(), [(0, 10), (20, 30)])
    assert manager.get_allocated_intervals() == []
    assert manager.get_total_available_length() == 20
    assert manager.get_summary().total_free_length == 20
    stats = manager.get_availability_stats()
    assert stats["total_free"] == 20
    assert stats["free_chunks"] == 2

    manager.reserve_interval(0, 30)
    missing_results = (
        manager.find_interval(0, 1),
        manager.find_best_fit(1, prefer_early=True),
        manager.find_best_fit(1, prefer_early=False),
        manager.find_largest_available(),
    )
    _expect_equal(missing_results, (None, None, None, None))


def test_mixed_auto_routing_validation_and_environment(monkeypatch) -> None:
    monkeypatch.setattr(mixed, "MetalBoundarySummaryManager", FakeMetalManager)
    monkeypatch.setenv("TREEMENDOUS_MIXED_SUMMARY_PATH", "auto")
    monkeypatch.setenv("TREEMENDOUS_MIXED_BEST_FIT_PATH", "auto")
    monkeypatch.setenv("TREEMENDOUS_MIXED_BEST_FIT_MIN_INTERVALS", "1")
    monkeypatch.setenv("TREEMENDOUS_MIXED_SUMMARY_MIN_INTERVALS", "1")
    monkeypatch.setenv("TREEMENDOUS_MIXED_SYNC_CPU", "yes")
    monkeypatch.setenv("TREEMENDOUS_MIXED_TRACK_ALLOCATIONS", "off")
    manager = mixed.MixedBoundarySummaryManager()
    manager.release_interval(0, 10)
    assert manager._should_use_gpu_best_fit()
    assert manager._should_use_gpu_summary()

    manager.best_fit_path = "cpu"
    manager.summary_path = "cpu"
    assert not manager._should_use_gpu_best_fit()
    assert not manager._should_use_gpu_summary()
    manager.best_fit_path = "gpu"
    manager.summary_path = "gpu"
    assert manager._should_use_gpu_best_fit()
    assert manager._should_use_gpu_summary()

    with pytest.raises(ValueError, match="summary_path"):
        mixed.MixedBoundarySummaryManager(summary_path="invalid")
    with pytest.raises(ValueError, match="best_fit_path"):
        mixed.MixedBoundarySummaryManager(best_fit_path="invalid")
    with pytest.raises(OverflowError, match="32-bit"):
        manager.release_interval(-(2**31) - 1, 0)
    with pytest.raises(ValueError, match="start < end"):
        manager.batch_release([(0, 0)])

    monkeypatch.setenv("BROKEN_INTEGER", "many")
    with pytest.raises(ValueError, match="must be an integer"):
        mixed._read_int_env("BROKEN_INTEGER", 2)
    monkeypatch.delenv("BROKEN_INTEGER")
    assert mixed._read_int_env("BROKEN_INTEGER", 2) == 2
    monkeypatch.setenv("EMPTY_INTEGER", "")
    assert mixed._read_int_env("EMPTY_INTEGER", 3) == 3
    monkeypatch.delenv("MISSING_BOOLEAN", raising=False)
    assert mixed._read_bool_env("MISSING_BOOLEAN", True)
    monkeypatch.setenv("MISSING_BOOLEAN", "TRUE")
    assert mixed._read_bool_env("MISSING_BOOLEAN", False)


def test_mixed_summary_conversion_handles_native_sentinels() -> None:
    assert (
        mixed.MixedBoundarySummaryManager._summary_from_metal(
            SimpleNamespace(interval_count=0)
        )
        == BoundarySummary.empty()
    )
    summary = mixed.MixedBoundarySummaryManager._summary_from_metal(
        SimpleNamespace(
            total_free_length=10,
            total_occupied_length=5,
            interval_count=2,
            largest_interval_length=6,
            largest_interval_start=-1,
            smallest_interval_length=-1,
            avg_interval_length=5.0,
            total_gaps=1,
            avg_gap_size=2.0,
            fragmentation_index=0.4,
            earliest_start=-1,
            latest_end=-1,
            utilization=1 / 3,
        )
    )
    assert summary.smallest_interval_length == 0
    sentinel_fields = (
        summary.largest_interval_start,
        summary.earliest_start,
        summary.latest_end,
    )
    _expect_equal(sentinel_fields, (None, None, None))


def test_mixed_requires_native_extension(monkeypatch) -> None:
    monkeypatch.setattr(mixed, "MetalBoundarySummaryManager", None)
    with pytest.raises(ImportError, match="Metal acceleration not available"):
        mixed.MixedBoundarySummaryManager()


def test_boundary_summary_demo_remains_callable_without_import_side_effects(
    capsys,
) -> None:
    assert capsys.readouterr().out == ""
    with pytest.warns(DeprecationWarning, match="deprecated"):
        demo_boundary_summary_performance()
    output = capsys.readouterr().out
    assert "Boundary Summary Manager Performance Demo" in output
    assert "Operations: 7" in output
