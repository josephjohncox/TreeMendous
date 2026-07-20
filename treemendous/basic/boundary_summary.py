"""Boundary-map interval manager with lazily cached summary statistics.

The first summary read after an effective mutation scans all free intervals;
subsequent reads are constant-time cache hits until the next mutation.
"""

from dataclasses import dataclass, replace
from typing import Any

from sortedcontainers import SortedDict

from treemendous.domain import (
    IntervalResult,
    ManagedDomain,
    Span,
    validate_coordinate,
    validate_length,
)


@dataclass(frozen=True)
class BoundaryPerformanceStats:
    operation_count: int
    cache_hits: int

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.operation_count if self.operation_count else 0.0


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
        managed_domain: ManagedDomain | Span | tuple[int, int] | None = None,
    ) -> None:
        self.intervals: SortedDict[int, tuple[int, None]] = SortedDict()

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
        self._managed_spans: SortedDict[int, int] | None = (
            None if self._domain_explicit else SortedDict()
        )

        self._operation_count = 0
        self._cache_hits = 0

    def _extend_inferred_domain(self, span: Span) -> None:
        managed = self._managed_spans
        if managed is None:
            raise RuntimeError("explicit domains cannot be extended")
        start = span.start
        end = span.end
        index = managed.bisect_left(start)
        if index > 0:
            previous_start = managed.keys()[index - 1]
            previous_end = managed[previous_start]
            if end <= previous_end:
                return
            if previous_end >= start:
                start = previous_start
                end = max(end, previous_end)
                index -= 1
                del managed[previous_start]
        while index < len(managed):
            current_start = managed.keys()[index]
            if current_start > end:
                break
            end = max(end, managed[current_start])
            del managed[current_start]
        managed[start] = end
        self._managed_domain = None

    def _current_managed_domain(self) -> ManagedDomain | None:
        if self._domain_explicit:
            return self._managed_domain
        managed = self._managed_spans
        if managed is None:
            raise RuntimeError("inferred domain state is unavailable")
        if not managed:
            return None
        if self._managed_domain is None:
            self._managed_domain = ManagedDomain(
                Span(start, end) for start, end in managed.items()
            )
        return self._managed_domain

    def release_interval(self, start: int, end: int) -> None:
        """Add interval to available space with summary update"""
        span = Span(start, end)
        if self._domain_explicit:
            assert self._managed_domain is not None
            if not self._managed_domain.contains(span):
                raise ValueError("span must be contained in the managed domain")
        self._operation_count += 1

        # Find insertion position
        idx = self.intervals.bisect_left(start)

        # A canonical interval containing the release makes it a no-op.
        if idx > 0:
            prev_start = self.intervals.keys()[idx - 1]
            prev_end, _ = self.intervals[prev_start]
            if end <= prev_end:
                return
        if idx < len(self.intervals):
            current_start = self.intervals.keys()[idx]
            current_end, _ = self.intervals[current_start]
            if current_start == start and end <= current_end:
                return

        # Merge with the previous interval if overlapping or adjacent.
        if idx > 0:
            prev_start = self.intervals.keys()[idx - 1]
            prev_end, _ = self.intervals[prev_start]
            if prev_end >= start:
                start = prev_start
                end = max(end, prev_end)
                idx -= 1
                del self.intervals[prev_start]

        # Merge with following intervals if overlapping
        while idx < len(self.intervals):
            curr_start = self.intervals.keys()[idx]
            curr_end, _ = self.intervals[curr_start]
            if curr_start > end:
                break
            end = max(end, curr_end)
            del self.intervals[curr_start]

        # Insert merged interval
        self.intervals[start] = (end, None)
        if not self._domain_explicit:
            self._extend_inferred_domain(span)
        self._summary_dirty = True

    def reserve_interval(self, start: int, end: int) -> None:
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

        intervals_to_add: list[tuple[int, int]] = []
        keys_to_delete = []

        keys = self.intervals.keys()
        while idx < len(keys):
            curr_start = keys[idx]
            curr_end, _ = self.intervals[curr_start]

            if curr_start >= end:
                break

            overlap_start = max(start, curr_start)
            overlap_end = min(end, curr_end)

            if overlap_start < overlap_end:
                keys_to_delete.append(curr_start)

                # Retain the non-overlapping geometry.
                if curr_start < start:
                    intervals_to_add.append((curr_start, start))
                if curr_end > end:
                    intervals_to_add.append((end, curr_end))

            idx += 1

        # Apply deletions and additions
        for key in keys_to_delete:
            del self.intervals[key]

        for remaining_start, remaining_end in intervals_to_add:
            self.intervals[remaining_start] = (remaining_end, None)
        if keys_to_delete:
            self._summary_dirty = True

    def find_interval(self, start: int, length: int) -> IntervalResult | None:
        """Return the earliest fit at or after *start*."""
        validate_coordinate(start, "start")
        validate_length(length)
        idx = self.intervals.bisect_right(start) - 1
        if idx >= 0:
            interval_start = self.intervals.keys()[idx]
            interval_end, _ = self.intervals[interval_start]
            allocation_start = max(start, interval_start)
            if allocation_start + length <= interval_end:
                return IntervalResult(allocation_start, allocation_start + length)
            idx += 1
        else:
            idx = 0
        while idx < len(self.intervals):
            interval_start = self.intervals.keys()[idx]
            interval_end, _ = self.intervals[interval_start]
            allocation_start = max(start, interval_start)
            if allocation_start + length <= interval_end:
                return IntervalResult(allocation_start, allocation_start + length)
            idx += 1
        return None

    def get_intervals(self) -> list[IntervalResult]:
        """Get all available intervals"""
        return [
            IntervalResult(start=start, end=end)
            for start, (end, _) in self.intervals.items()
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
        managed_domain = self._current_managed_domain()
        if managed_domain is not None:
            occupied = managed_domain.measure - summary.total_free_length
            bounds = managed_domain.bounds
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
                utilization=occupied / managed_domain.measure,
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
            (start, end)
            for start, (end, _) in self.intervals.items()
            if end - start >= length
        ]
        if not candidates:
            return None
        if prefer_early:
            start, _ = min(candidates, key=lambda item: item[0])
        else:
            start, _ = min(candidates, key=lambda item: (item[1] - item[0], item[0]))
        return IntervalResult(start=start, end=start + length)

    def find_largest_available(self) -> IntervalResult | None:
        """Find largest available interval using summary optimization"""
        summary = self.get_summary()

        if summary.largest_interval_length == 0:
            return None

        # Find the interval with largest size
        for start, (end, _) in self.intervals.items():
            if (end - start) == summary.largest_interval_length:
                return IntervalResult(start=start, end=end)

        return None

    def get_performance_stats(self) -> BoundaryPerformanceStats:
        """Return cache counters for this raw backend."""
        return BoundaryPerformanceStats(
            operation_count=self._operation_count,
            cache_hits=self._cache_hits,
        )

    def print_intervals(self) -> None:
        """Print intervals with summary information"""
        print("Boundary-Based Summary Interval Manager:")
        print(f"Available intervals ({len(self.intervals)}):")

        for start, (end, _) in self.intervals.items():
            print(f"  [{start}, {end}) length={end - start}")

        summary = self.get_summary()
        print("\nSummary Statistics:")
        print(f"  Total free: {summary.total_free_length}")
        print(f"  Intervals: {summary.interval_count}")
        print(f"  Largest: {summary.largest_interval_length}")
        print(f"  Fragmentation: {summary.fragmentation_index:.2f}")
        print(f"  Utilization: {summary.utilization:.2%}")

        perf = self.get_performance_stats()
        print(f"  Cache hit rate: {perf.cache_hit_rate:.1%}")
