"""
Boundary-Based Summary Interval Tree

Combines the simplicity and efficiency of boundary management (SortedDict)
with comprehensive summary statistics for O(1) analytics. This hybrid approach
provides the best of both worlds: simple implementation with advanced analytics.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from sortedcontainers import SortedDict

from treemendous.basic.protocols import IntervalResult, PerformanceStats
from treemendous.domain import ManagedDomain, Span, validate_coordinate, validate_length


@dataclass
class BoundarySummary:
    """Summary statistics optimized for boundary-based interval management"""

    # Core metrics
    total_free_length: int = 0
    total_occupied_length: int = 0
    interval_count: int = 0

    # Efficiency metrics
    largest_interval_length: int = 0
    largest_interval_start: int | None = None
    smallest_interval_length: int = 0
    avg_interval_length: float = 0.0

    # Space distribution
    total_gaps: int = 0  # Number of gaps between intervals
    avg_gap_size: float = 0.0
    fragmentation_index: float = 0.0  # 1 - (largest / total)

    # Bounds
    earliest_start: int | None = None
    latest_end: int | None = None

    # Utilization (requires managed space tracking)
    utilization: float = 0.0

    @classmethod
    def empty(cls) -> "BoundarySummary":
        """Create empty summary"""
        return cls()

    @classmethod
    def compute_from_intervals(
        cls,
        intervals: SortedDict,
        managed_start: int | None = None,
        managed_end: int | None = None,
    ) -> "BoundarySummary":
        """Compute summary statistics from boundary intervals"""
        if not intervals:
            if managed_start is not None and managed_end is not None:
                managed_space = managed_end - managed_start
                return cls(
                    total_occupied_length=managed_space,
                    earliest_start=managed_start,
                    latest_end=managed_end,
                    utilization=1.0 if managed_space else 0.0,
                )
            return cls.empty()

        # Basic interval statistics
        interval_count = len(intervals)
        total_free = sum(end - start for start, (end, _) in intervals.items())

        # Find largest and smallest intervals
        interval_lengths = [end - start for start, (end, _) in intervals.items()]
        largest_length = max(interval_lengths)
        smallest_length = min(interval_lengths)
        avg_length = total_free / interval_count if interval_count > 0 else 0.0

        # Find largest interval start
        largest_start = None
        for start, (end, _) in intervals.items():
            if end - start == largest_length:
                largest_start = start
                break

        # Calculate gaps between intervals
        sorted_intervals = list(intervals.items())
        gaps = []

        for i in range(len(sorted_intervals) - 1):
            current_end = sorted_intervals[i][1][0]
            next_start = sorted_intervals[i + 1][0]
            if next_start > current_end:
                gaps.append(next_start - current_end)

        total_gaps = len(gaps)
        avg_gap_size = sum(gaps) / len(gaps) if gaps else 0.0

        # Fragmentation index
        fragmentation = 1.0 - (largest_length / total_free) if total_free > 0 else 0.0

        # Bounds
        earliest_start = min(intervals.keys()) if intervals else None
        latest_end = max(end for end, _ in intervals.values()) if intervals else None

        # Utilization calculation
        utilization = 0.0
        total_occupied = 0

        if managed_start is not None and managed_end is not None:
            managed_space = managed_end - managed_start
            total_occupied = managed_space - total_free
            utilization = total_occupied / managed_space if managed_space > 0 else 0.0

        return cls(
            total_free_length=total_free,
            total_occupied_length=total_occupied,
            interval_count=interval_count,
            largest_interval_length=largest_length,
            largest_interval_start=largest_start,
            smallest_interval_length=smallest_length,
            avg_interval_length=avg_length,
            total_gaps=total_gaps,
            avg_gap_size=avg_gap_size,
            fragmentation_index=fragmentation,
            earliest_start=earliest_start,
            latest_end=latest_end,
            utilization=utilization,
        )


class BoundarySummaryManager:
    """Boundary manager enhanced with comprehensive summary statistics"""

    def __init__(
        self,
        merge_fn: Callable[[Any, Any], Any] | None = None,
        split_fn: Callable[[Any, int, int, int, int], Any] | None = None,
        can_merge: Callable[[Any | None, Any | None], bool] | None = None,
        merge_idempotent: bool = False,
        split_idempotent: bool = False,
        managed_domain: ManagedDomain | Span | tuple[int, int] | None = None,
    ):
        # Core boundary management
        self.intervals: SortedDict[int, tuple[int, Any | None]] = SortedDict()

        # Summary statistics caching
        self._cached_summary: BoundarySummary | None = None
        self._summary_dirty = True

        # Analytics use a normalized domain, never a convex inferred hull.
        self._managed_domain = (
            managed_domain
            if isinstance(managed_domain, ManagedDomain)
            else (ManagedDomain(managed_domain) if managed_domain is not None else None)
        )
        self._domain_explicit = managed_domain is not None

        # Performance tracking
        self._operation_count = 0
        self._cache_hits = 0
        self._merge_fn = merge_fn
        self._split_fn = split_fn
        self._can_merge = can_merge
        self._merge_idempotent = merge_idempotent
        self._split_idempotent = split_idempotent

    def _merge_data(self, data1: Any | None, data2: Any | None) -> Any | None:
        if data1 is None:
            return data2
        if data2 is None:
            return data1
        if self._merge_idempotent and (data1 is data2 or data1 == data2):
            return data1
        if self._merge_fn is None:
            if isinstance(data1, set) and isinstance(data2, set):
                return data1 | data2
            return data1
        return self._merge_fn(data1, data2)

    def _split_data(
        self,
        data: Any | None,
        old_start: int,
        old_end: int,
        new_start: int,
        new_end: int,
    ) -> Any | None:
        if data is None:
            return None
        if self._split_fn is None:
            return data
        return self._split_fn(data, old_start, old_end, new_start, new_end)

    def _can_merge_data(self, data1: Any | None, data2: Any | None) -> bool:
        if self._can_merge is None:
            return True
        return self._can_merge(data1, data2)

    def release_interval(self, start: int, end: int, data: Any | None = None) -> None:
        """Add interval to available space with summary update"""
        span = Span(start, end)
        if self._domain_explicit:
            assert self._managed_domain is not None
            if not self._managed_domain.contains(span):
                raise ValueError("span must be contained in the managed domain")
        before = tuple(self.intervals.items())
        self._operation_count += 1

        # Find insertion position
        idx = self.intervals.bisect_left(start)
        merged_data = data

        # Merge with previous interval if overlapping or adjacent
        if idx > 0:
            prev_start = self.intervals.keys()[idx - 1]
            prev_end, prev_data = self.intervals[prev_start]
            if prev_end > start or (
                prev_end == start and self._can_merge_data(prev_data, merged_data)
            ):
                start = prev_start
                end = max(end, prev_end)
                merged_data = self._merge_data(merged_data, prev_data)
                idx -= 1
                del self.intervals[prev_start]

        # Merge with following intervals if overlapping
        while idx < len(self.intervals):
            curr_start = self.intervals.keys()[idx]
            curr_end, curr_data = self.intervals[curr_start]
            if curr_start > end:
                break
            if curr_start == end and not self._can_merge_data(merged_data, curr_data):
                break
            end = max(end, curr_end)
            merged_data = self._merge_data(merged_data, curr_data)
            del self.intervals[curr_start]

        # Insert merged interval
        self.intervals[start] = (end, merged_data)
        if not self._domain_explicit:
            self._managed_domain = (
                self._managed_domain.extended(span)
                if self._managed_domain is not None
                else ManagedDomain(span)
            )
        if tuple(self.intervals.items()) != before:
            self._summary_dirty = True

    def reserve_interval(self, start: int, end: int, data: Any | None = None) -> None:
        """Remove interval from available space with summary update"""
        span = Span(start, end)
        if self._domain_explicit:
            assert self._managed_domain is not None
            if not self._managed_domain.contains(span):
                raise ValueError("span must be contained in the managed domain")
        self._operation_count += 1

        idx = self.intervals.bisect_left(start)

        if idx > 0:
            prev_start = self.intervals.keys()[idx - 1]
            prev_end, _ = self.intervals[prev_start]
            if prev_end > start:
                idx -= 1

        intervals_to_add: list[tuple[int, int, Any | None]] = []
        keys_to_delete = []

        keys = list(self.intervals.keys())
        while idx < len(keys):
            curr_start = keys[idx]
            curr_end, curr_data = self.intervals[curr_start]

            if curr_start >= end:
                break

            overlap_start = max(start, curr_start)
            overlap_end = min(end, curr_end)

            if overlap_start < overlap_end:
                keys_to_delete.append(curr_start)

                # Add non-overlapping parts
                if curr_start < start:
                    left_data = self._split_data(
                        curr_data, curr_start, curr_end, curr_start, start
                    )
                    intervals_to_add.append((curr_start, start, left_data))
                if curr_end > end:
                    right_data = self._split_data(
                        curr_data, curr_start, curr_end, end, curr_end
                    )
                    intervals_to_add.append((end, curr_end, right_data))

            idx += 1

        # Apply deletions and additions
        for key in keys_to_delete:
            del self.intervals[key]

        for s, e, interval_data in intervals_to_add:
            self.intervals[s] = (e, interval_data)
        if keys_to_delete:
            self._summary_dirty = True

    def find_interval(self, start: int, length: int) -> IntervalResult | None:
        """Return the earliest fit at or after *start*."""
        validate_coordinate(start, "start")
        validate_length(length)
        idx = self.intervals.bisect_right(start) - 1
        if idx >= 0:
            s = self.intervals.keys()[idx]
            e, data = self.intervals[s]
            allocation_start = max(start, s)
            if allocation_start + length <= e:
                return IntervalResult(
                    allocation_start, allocation_start + length, data=data
                )
            idx += 1
        else:
            idx = 0
        while idx < len(self.intervals):
            s = self.intervals.keys()[idx]
            e, data = self.intervals[s]
            allocation_start = max(start, s)
            if allocation_start + length <= e:
                return IntervalResult(
                    allocation_start, allocation_start + length, data=data
                )
            idx += 1
        return None

    def get_intervals(self) -> list[IntervalResult]:
        """Get all available intervals"""
        return [
            IntervalResult(start=start, end=end, data=data)
            for start, (end, data) in self.intervals.items()
        ]

    def get_total_available_length(self) -> int:
        """Get total available space (with caching)"""
        summary = self.get_summary()
        return summary.total_free_length

    def get_summary(self) -> BoundarySummary:
        """Get comprehensive summary statistics (cached for performance)"""
        if not self._summary_dirty and self._cached_summary is not None:
            self._cache_hits += 1
            return self._cached_summary

        # Recompute primitive geometry, then derive occupancy from the normalized
        # domain measure. Disjoint unmanaged gaps are never counted as occupied.
        summary = BoundarySummary.compute_from_intervals(self.intervals)
        if self._managed_domain is not None:
            occupied = self._managed_domain.measure - summary.total_free_length
            bounds = self._managed_domain.bounds
            summary = replace(
                summary,
                total_occupied_length=occupied,
                earliest_start=(
                    summary.earliest_start
                    if summary.earliest_start is not None
                    else bounds[0]
                ),
                latest_end=(
                    summary.latest_end if summary.latest_end is not None else bounds[1]
                ),
                utilization=occupied / self._managed_domain.measure,
            )
        self._cached_summary = summary
        self._summary_dirty = False

        return self._cached_summary

    def get_availability_stats(self) -> dict[str, Any]:
        """Get availability statistics in standard format"""
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
            "free_density": (
                summary.total_free_length
                / (summary.total_free_length + summary.total_occupied_length)
                if summary.total_free_length + summary.total_occupied_length
                else 0.0
            ),
            "bounds": (summary.earliest_start, summary.latest_end),
            "gaps": summary.total_gaps,
            "avg_gap_size": summary.avg_gap_size,
        }

    def find_best_fit(
        self, length: int, prefer_early: bool = True
    ) -> IntervalResult | None:
        """Find earliest fit or least-waste fit with deterministic ties."""
        validate_length(length)
        candidates = [
            (start, end, data)
            for start, (end, data) in self.intervals.items()
            if end - start >= length
        ]
        if not candidates:
            return None
        if prefer_early:
            start, _, data = min(candidates, key=lambda item: item[0])
        else:
            start, end, data = min(
                candidates, key=lambda item: (item[1] - item[0], item[0])
            )
        return IntervalResult(start=start, end=start + length, data=data)

    def find_largest_available(self) -> IntervalResult | None:
        """Find largest available interval using summary optimization"""
        summary = self.get_summary()

        if summary.largest_interval_length == 0:
            return None

        # Find the interval with largest size
        for start, (end, data) in self.intervals.items():
            if (end - start) == summary.largest_interval_length:
                return IntervalResult(start=start, end=end, data=data)

        return None

    def get_performance_stats(self) -> PerformanceStats:
        """Get implementation-specific performance statistics"""
        return PerformanceStats(
            operation_count=self._operation_count,
            cache_hits=self._cache_hits,
            cache_hit_rate=self._cache_hits / max(1, self._operation_count),
            implementation_name="boundary_summary",
            language="Python",
        )

    def print_intervals(self) -> None:
        """Print intervals with summary information"""
        print("Boundary-Based Summary Interval Manager:")
        print(f"Available intervals ({len(self.intervals)}):")

        for start, (end, data) in self.intervals.items():
            suffix = f" data={data}" if data is not None else ""
            print(f"  [{start}, {end}) length={end - start}{suffix}")

        summary = self.get_summary()
        print("\nSummary Statistics:")
        print(f"  Total free: {summary.total_free_length}")
        print(f"  Intervals: {summary.interval_count}")
        print(f"  Largest: {summary.largest_interval_length}")
        print(f"  Fragmentation: {summary.fragmentation_index:.2f}")
        print(f"  Utilization: {summary.utilization:.2%}")

        perf = self.get_performance_stats()
        print(f"  Cache hit rate: {perf.cache_hit_rate:.1%}")


# Convenience functions for different use cases
def create_boundary_summary_manager() -> BoundarySummaryManager:
    """Create boundary summary manager with optimal configuration"""
    return BoundarySummaryManager()


def demo_boundary_summary_performance() -> None:
    """Run the deprecated, deterministic boundary-summary console demo.

    The historical function remains callable for compatibility, but importing
    this module never runs it. Use the validated performance harness for timing.
    """
    warnings.warn(
        "demo_boundary_summary_performance is deprecated; use the validated "
        "performance harness instead",
        DeprecationWarning,
        stacklevel=2,
    )
    manager = BoundarySummaryManager()
    manager.release_interval(0, 10_000)
    for operation, start, end in (
        ("reserve", 1_000, 1_500),
        ("reserve", 3_000, 3_200),
        ("reserve", 5_000, 5_800),
        ("release", 2_000, 2_500),
        ("reserve", 7_000, 7_300),
        ("release", 6_000, 6_500),
    ):
        if operation == "reserve":
            manager.reserve_interval(start, end)
        else:
            manager.release_interval(start, end)

    summary = manager.get_summary()
    performance = manager.get_performance_stats()
    print("Boundary Summary Manager Performance Demo")
    print(f"Total free: {summary.total_free_length}")
    print(f"Intervals: {summary.interval_count}")
    print(f"Operations: {performance.operation_count}")
