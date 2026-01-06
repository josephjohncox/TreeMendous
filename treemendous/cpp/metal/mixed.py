"""
Mixed Metal/CPU Boundary Summary Manager

Routes latency-sensitive operations to the fastest path:
- CPU (cached) for summary/availability stats
- Metal GPU for best-fit queries when enabled

Maintains a CPU boundary summary manager for cached stats and a Metal manager
for GPU acceleration. Payloads for reserved intervals can be tracked in a
separate allocated-interval store when track_allocations=True. Interval updates
are applied to both unless sync_cpu=False.
"""

import os
from typing import Optional, Any, Dict

from treemendous.basic.boundary_summary import BoundarySummaryManager, BoundarySummary, IntervalResult
from treemendous.basic.boundary import IntervalManager as AllocationIntervalManager

try:
    from treemendous.cpp.metal.boundary_summary_metal import MetalBoundarySummaryManager
except ImportError as e:  # pragma: no cover - import guarded by availability checks
    MetalBoundarySummaryManager = None
    _import_error = e


class MixedBoundarySummaryManager:
    """Mixed CPU/Metal manager with configurable fast paths."""

    def __init__(
        self,
        summary_path: Optional[str] = None,
        best_fit_path: Optional[str] = None,
        best_fit_min_intervals: Optional[int] = None,
        summary_min_intervals: Optional[int] = None,
        sync_cpu: Optional[bool] = None,
        track_allocations: Optional[bool] = None,
    ) -> None:
        if MetalBoundarySummaryManager is None:
            raise ImportError(
                "Metal acceleration not available. "
                "Build with: python setup_metal.py build_ext --inplace"
            )

        self.summary_path = summary_path or os.environ.get(
            "TREEMENDOUS_MIXED_SUMMARY_PATH", "cpu"
        )
        self.best_fit_path = best_fit_path or os.environ.get(
            "TREEMENDOUS_MIXED_BEST_FIT_PATH", "auto"
        )
        self.best_fit_min_intervals = (
            best_fit_min_intervals
            if best_fit_min_intervals is not None
            else _read_int_env("TREEMENDOUS_MIXED_BEST_FIT_MIN_INTERVALS", 10_000)
        )
        self.summary_min_intervals = (
            summary_min_intervals
            if summary_min_intervals is not None
            else _read_int_env("TREEMENDOUS_MIXED_SUMMARY_MIN_INTERVALS", 50_000)
        )
        self.sync_cpu = (
            sync_cpu
            if sync_cpu is not None
            else _read_bool_env("TREEMENDOUS_MIXED_SYNC_CPU", True)
        )
        self.track_allocations = (
            track_allocations
            if track_allocations is not None
            else _read_bool_env("TREEMENDOUS_MIXED_TRACK_ALLOCATIONS", True)
        )

        if not self.sync_cpu:
            if self.summary_path == "cpu":
                self.summary_path = "gpu"
            if self.best_fit_path == "cpu":
                self.best_fit_path = "gpu"

        self.cpu_manager = BoundarySummaryManager()
        self.metal_manager = MetalBoundarySummaryManager()
        self.allocated_manager = (
            AllocationIntervalManager(can_merge=lambda a, b: a == b)
            if self.track_allocations
            else None
        )

        if self.summary_path not in {"cpu", "gpu", "auto"}:
            raise ValueError("summary_path must be 'cpu', 'gpu', or 'auto'")
        if self.best_fit_path not in {"cpu", "gpu", "auto"}:
            raise ValueError("best_fit_path must be 'cpu', 'gpu', or 'auto'")

    def _interval_count(self) -> int:
        if self.sync_cpu:
            return len(self.cpu_manager.intervals)
        return 0

    def _should_use_gpu_best_fit(self) -> bool:
        if not self.sync_cpu:
            return True
        if self.best_fit_path == "gpu":
            return True
        if self.best_fit_path == "cpu":
            return False
        return self._interval_count() >= self.best_fit_min_intervals

    def _should_use_gpu_summary(self) -> bool:
        if not self.sync_cpu:
            return True
        if self.summary_path == "gpu":
            return True
        if self.summary_path == "cpu":
            return False
        return self._interval_count() >= self.summary_min_intervals

    @staticmethod
    def _summary_from_metal(summary) -> BoundarySummary:
        if summary.interval_count == 0:
            return BoundarySummary.empty()

        smallest = summary.smallest_interval_length
        if smallest is None or smallest < 0:
            smallest = 0

        largest_start = summary.largest_interval_start
        if largest_start is not None and largest_start < 0:
            largest_start = None

        earliest = summary.earliest_start
        if earliest is not None and earliest < 0:
            earliest = None

        latest = summary.latest_end
        if latest is not None and latest < 0:
            latest = None

        return BoundarySummary(
            total_free_length=summary.total_free_length,
            total_occupied_length=summary.total_occupied_length,
            interval_count=summary.interval_count,
            largest_interval_length=summary.largest_interval_length,
            largest_interval_start=largest_start,
            smallest_interval_length=smallest,
            avg_interval_length=summary.avg_interval_length,
            total_gaps=summary.total_gaps,
            avg_gap_size=summary.avg_gap_size,
            fragmentation_index=summary.fragmentation_index,
            earliest_start=earliest,
            latest_end=latest,
            utilization=summary.utilization,
        )

    # Core interval operations
    def release_interval(self, start: int, end: int, data: Optional[Any] = None) -> None:
        if self.sync_cpu:
            self.cpu_manager.release_interval(start, end, data)
        self.metal_manager.release_interval(start, end)
        if self.allocated_manager is not None:
            self.allocated_manager.reserve_interval(start, end)

    def reserve_interval(self, start: int, end: int, data: Optional[Any] = None) -> None:
        if self.sync_cpu:
            self.cpu_manager.reserve_interval(start, end)
        self.metal_manager.reserve_interval(start, end)
        if self.allocated_manager is not None:
            self.allocated_manager.release_interval(start, end, data)

    def batch_reserve(self, intervals) -> None:
        if self.sync_cpu:
            for start, end in intervals:
                self.cpu_manager.reserve_interval(start, end)
        self.metal_manager.batch_reserve(intervals)
        if self.allocated_manager is not None:
            for start, end in intervals:
                self.allocated_manager.release_interval(start, end)

    def batch_release(self, intervals) -> None:
        if self.sync_cpu:
            for start, end in intervals:
                self.cpu_manager.release_interval(start, end)
        self.metal_manager.batch_release(intervals)
        if self.allocated_manager is not None:
            for start, end in intervals:
                self.allocated_manager.reserve_interval(start, end)

    # Queries
    def find_interval(self, start: int, length: int) -> Optional[IntervalResult]:
        if self.sync_cpu:
            return self.cpu_manager.find_interval(start, length)
        result = self.metal_manager.find_interval(start, length)
        if result is None:
            return None
        if isinstance(result, tuple):
            return IntervalResult(start=result[0], end=result[1])
        return result

    def find_best_fit(self, length: int, prefer_early: bool = True):
        if prefer_early:
            if self.sync_cpu:
                return self.cpu_manager.find_best_fit(length, prefer_early=True)
            result = self.metal_manager.find_best_fit(length, prefer_early=True)
            if result is None:
                return None
            if isinstance(result, tuple):
                return IntervalResult(start=result[0], end=result[1])
            return result

        if self._should_use_gpu_best_fit():
            result = self.metal_manager.find_best_fit(length, prefer_early=False)
            if result is None:
                return None
            if isinstance(result, tuple):
                return IntervalResult(start=result[0], end=result[1])
            return result

        if self.sync_cpu:
            return self.cpu_manager.find_best_fit(length, prefer_early=False)
        return None

    def find_largest_available(self):
        if self.sync_cpu:
            return self.cpu_manager.find_largest_available()
        result = self.metal_manager.find_largest_available()
        if result is None:
            return None
        if isinstance(result, tuple):
            return IntervalResult(start=result[0], end=result[1])
        return result

    def get_intervals(self):
        if self.sync_cpu:
            return self.cpu_manager.get_intervals()
        return self.metal_manager.get_intervals()

    def get_allocated_intervals(self):
        if self.allocated_manager is None:
            return []
        return self.allocated_manager.get_intervals()

    def get_total_available_length(self) -> int:
        if self._should_use_gpu_summary():
            return self.metal_manager.compute_summary_gpu().total_free_length
        return self.cpu_manager.get_total_available_length()

    def get_summary(self) -> BoundarySummary:
        if self._should_use_gpu_summary():
            return self._summary_from_metal(self.metal_manager.compute_summary_gpu())
        return self.cpu_manager.get_summary()

    def get_availability_stats(self) -> Dict[str, Any]:
        if not self._should_use_gpu_summary():
            return self.cpu_manager.get_availability_stats()

        summary = self.get_summary()
        return {
            "total_free": summary.total_free_length,
            "total_occupied": summary.total_occupied_length,
            "total_space": summary.total_free_length + summary.total_occupied_length,
            "free_chunks": summary.interval_count,
            "largest_chunk": summary.largest_interval_length,
            "avg_chunk_size": summary.avg_interval_length,
            "utilization": summary.utilization,
            "fragmentation": summary.fragmentation_index,
            "free_density": 1.0 - summary.utilization,
            "bounds": (summary.earliest_start, summary.latest_end),
            "gaps": summary.total_gaps,
            "avg_gap_size": summary.avg_gap_size,
        }

    # Metal passthroughs
    def compute_summary_gpu(self):
        return self.metal_manager.compute_summary_gpu()

    def compute_summary_cpu(self):
        return self.metal_manager.compute_summary_cpu()

    def get_performance_stats(self):
        return self.metal_manager.get_performance_stats()

    def print_info(self) -> None:
        self.metal_manager.print_info()


def _read_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer (got '{value}')") from exc


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
