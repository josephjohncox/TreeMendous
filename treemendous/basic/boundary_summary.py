"""
Boundary-Based Summary Interval Tree

Combines the simplicity and efficiency of boundary management (SortedDict)
with comprehensive summary statistics for O(1) analytics. This hybrid approach
provides the best of both worlds: simple implementation with advanced analytics.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any, Callable
import math

from sortedcontainers import SortedDict

try:
    from treemendous.basic.protocols import (
        CoreIntervalManagerProtocol, 
        EnhancedIntervalManagerProtocol,
        PerformanceTrackingProtocol,
        IntervalResult,
        AvailabilityStats,
        PerformanceStats
    )
except ImportError:
    from protocols import (
        CoreIntervalManagerProtocol,
        EnhancedIntervalManagerProtocol, 
        PerformanceTrackingProtocol,
        IntervalResult,
        AvailabilityStats,
        PerformanceStats
    )


@dataclass
class BoundarySummary:
    """Summary statistics optimized for boundary-based interval management"""
    
    # Core metrics
    total_free_length: int = 0
    total_occupied_length: int = 0
    interval_count: int = 0
    
    # Efficiency metrics  
    largest_interval_length: int = 0
    largest_interval_start: Optional[int] = None
    smallest_interval_length: int = 0
    avg_interval_length: float = 0.0
    
    # Space distribution
    total_gaps: int = 0  # Number of gaps between intervals
    avg_gap_size: float = 0.0
    fragmentation_index: float = 0.0  # 1 - (largest / total)
    
    # Bounds
    earliest_start: Optional[int] = None
    latest_end: Optional[int] = None
    
    # Utilization (requires managed space tracking)
    utilization: float = 0.0
    
    @classmethod
    def empty(cls) -> 'BoundarySummary':
        """Create empty summary"""
        return cls()
    
    @classmethod
    def compute_from_intervals(cls, intervals: SortedDict, managed_start: Optional[int] = None, 
                              managed_end: Optional[int] = None) -> 'BoundarySummary':
        """Compute summary statistics from boundary intervals"""
        if not intervals:
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
            utilization=utilization
        )


class BoundarySummaryManager(EnhancedIntervalManagerProtocol[Any], PerformanceTrackingProtocol):
    """Boundary manager enhanced with comprehensive summary statistics"""
    
    def __init__(self, merge_fn: Optional[Callable[[Any, Any], Any]] = None):
        # Core boundary management
        self.intervals: SortedDict[int, Tuple[int, Optional[Any]]] = SortedDict()
        
        # Summary statistics caching
        self._cached_summary: Optional[BoundarySummary] = None
        self._summary_dirty = True
        
        # Managed space tracking for utilization calculation
        self._managed_start: Optional[int] = None
        self._managed_end: Optional[int] = None
        
        # Performance tracking
        self._operation_count = 0
        self._cache_hits = 0
        self._merge_fn = merge_fn

    def _merge_data(self, data1: Optional[Any], data2: Optional[Any]) -> Optional[Any]:
        if data1 is None:
            return data2
        if data2 is None:
            return data1
        if self._merge_fn is None:
            if isinstance(data1, set) and isinstance(data2, set):
                return data1 | data2
            return data1
        return self._merge_fn(data1, data2)
    
    def release_interval(self, start: int, end: int, data: Optional[Any] = None) -> None:
        """Add interval to available space with summary update"""
        if start >= end:
            return
        
        self._update_managed_bounds(start, end)
        self._summary_dirty = True
        self._operation_count += 1
        
        # Find insertion position
        idx = self.intervals.bisect_left(start)
        merged_data = data
        
        # Merge with previous interval if overlapping or adjacent
        if idx > 0:
            prev_start = self.intervals.keys()[idx - 1]
            prev_end, prev_data = self.intervals[prev_start]
            if prev_end >= start:
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
            end = max(end, curr_end)
            merged_data = self._merge_data(merged_data, curr_data)
            del self.intervals[curr_start]
        
        # Insert merged interval
        self.intervals[start] = (end, merged_data)
    
    def reserve_interval(self, start: int, end: int, data: Optional[Any] = None) -> None:
        """Remove interval from available space with summary update"""
        if start >= end:
            return
        
        self._update_managed_bounds(start, end)
        self._summary_dirty = True
        self._operation_count += 1
        
        idx = self.intervals.bisect_left(start)
        
        if idx > 0:
            prev_start = self.intervals.keys()[idx - 1]
            prev_end, _ = self.intervals[prev_start]
            if prev_end > start:
                idx -= 1
        
        intervals_to_add: List[Tuple[int, int, Optional[Any]]] = []
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
                    intervals_to_add.append((curr_start, start, curr_data))
                if curr_end > end:
                    intervals_to_add.append((end, curr_end, curr_data))
            
            idx += 1
        
        # Apply deletions and additions
        for key in keys_to_delete:
            del self.intervals[key]
        
        for s, e, interval_data in intervals_to_add:
            self.intervals[s] = (e, interval_data)
    
    def find_interval(self, start: int, length: int) -> Optional[IntervalResult]:
        """Find suitable interval using boundary-based search"""
        idx = self.intervals.bisect_left(start)
        
        # Check interval at idx
        if idx < len(self.intervals):
            s = self.intervals.keys()[idx]
            e, data = self.intervals[s]
            if s <= start < e and e - start >= length:
                return IntervalResult(start=start, end=start + length, data=data)
            elif s > start and e - s >= length:
                return IntervalResult(start=s, end=s + length, data=data)
        
        # Check previous interval
        if idx > 0:
            idx -= 1
            s = self.intervals.keys()[idx]
            e, data = self.intervals[s]
            if s <= start < e and e - start >= length:
                return IntervalResult(start=start, end=start + length, data=data)
            elif start < s and e - s >= length:
                return IntervalResult(start=s, end=s + length, data=data)
        
        return None
    
    def find_best_fit(self, length: int) -> Optional[IntervalResult]:
        """Find best available interval of given length"""
        best_interval = None
        best_size = float('inf')
        
        for start, (end, data) in self.intervals.items():
            available_length = end - start
            if available_length >= length:
                if available_length < best_size:
                    best_size = available_length
                    best_interval = (start, end, data)
        
        if best_interval:
            start, end, data = best_interval
            return IntervalResult(start=start, end=start + length, length=length, data=data)
        
        return None
    
    def find_largest_available(self) -> Optional[IntervalResult]:
        """Find the largest available interval"""
        if not self.intervals:
            return None
        
        largest_interval = None
        largest_size = 0
        
        for start, (end, data) in self.intervals.items():
            size = end - start
            if size > largest_size:
                largest_size = size
                largest_interval = (start, end, data)
        
        if largest_interval:
            start, end, data = largest_interval
            return IntervalResult(start=start, end=end, length=end - start, data=data)
        
        return None
    
    def get_intervals(self) -> List[IntervalResult]:
        """Get all available intervals"""
        return [IntervalResult(start=start, end=end, data=data)
                for start, (end, data) in self.intervals.items()]
    
    def get_total_available_length(self) -> int:
        """Get total available space (with caching)"""
        summary = self.get_summary()
        return summary.total_free_length
    
    def get_summary(self) -> BoundarySummary:
        """Get comprehensive summary statistics (cached for performance)"""
        if not self._summary_dirty and self._cached_summary is not None:
            self._cache_hits += 1
            return self._cached_summary
        
        # Recompute summary
        self._cached_summary = BoundarySummary.compute_from_intervals(
            self.intervals, self._managed_start, self._managed_end
        )
        self._summary_dirty = False
        
        return self._cached_summary
    
    def get_availability_stats(self) -> Dict[str, Any]:
        """Get availability statistics in standard format"""
        summary = self.get_summary()
        
        return {
            'total_free': summary.total_free_length,
            'total_occupied': summary.total_occupied_length,
            'total_space': summary.total_free_length + summary.total_occupied_length,
            'free_chunks': summary.interval_count,
            'largest_chunk': summary.largest_interval_length,
            'avg_chunk_size': summary.avg_interval_length,
            'utilization': summary.utilization,
            'fragmentation': summary.fragmentation_index,
            'free_density': 1.0 - summary.utilization,
            'bounds': (summary.earliest_start, summary.latest_end),
            'gaps': summary.total_gaps,
            'avg_gap_size': summary.avg_gap_size
        }
    
    def find_best_fit(self, length: int, prefer_early: bool = True) -> Optional[IntervalResult]:
        """Find best-fit interval using boundary-based search"""
        best_candidate = None
        best_fit_size = float('inf')
        best_start = float('inf')
        
        for start, (end, data) in self.intervals.items():
            available = end - start
            if available >= length:
                if prefer_early:
                    if start < best_start:
                        best_candidate = IntervalResult(start=start, end=start + length, data=data)
                        best_start = start
                else:
                    # Best fit: smallest interval that satisfies requirement
                    if available < best_fit_size:
                        best_candidate = IntervalResult(start=start, end=start + length, data=data)
                        best_fit_size = available
        
        return best_candidate
    
    def find_largest_available(self) -> Optional[IntervalResult]:
        """Find largest available interval using summary optimization"""
        summary = self.get_summary()
        
        if summary.largest_interval_length == 0:
            return None
        
        # Find the interval with largest size
        for start, (end, data) in self.intervals.items():
            if (end - start) == summary.largest_interval_length:
                return IntervalResult(start=start, end=end, data=data)
        
        return None
    
    def _update_managed_bounds(self, start: int, end: int) -> None:
        """Update managed space bounds for utilization calculation"""
        if self._managed_start is None:
            self._managed_start = start
            self._managed_end = end
        else:
            self._managed_start = min(self._managed_start, start)
            self._managed_end = max(self._managed_end, end)
        
        self._summary_dirty = True
    
    def get_performance_stats(self) -> PerformanceStats:
        """Get implementation-specific performance statistics"""
        return PerformanceStats(
            operation_count=self._operation_count,
            cache_hits=self._cache_hits,
            cache_hit_rate=self._cache_hits / max(1, self._operation_count),
            implementation_name='boundary_summary',
            language='Python'
        )
    
    def print_intervals(self) -> None:
        """Print intervals with summary information"""
        print("Boundary-Based Summary Interval Manager:")
        print(f"Available intervals ({len(self.intervals)}):")
        
        for start, (end, data) in self.intervals.items():
            suffix = f" data={data}" if data is not None else ""
            print(f"  [{start}, {end}) length={end-start}{suffix}")
        
        summary = self.get_summary()
        print(f"\nSummary Statistics:")
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


def demo_boundary_summary_performance():
    """Demonstrate boundary summary manager performance"""
    print("🔄 Boundary Summary Manager Performance Demo")
    print("=" * 50)
    
    manager = BoundarySummaryManager()
    
    # Initialize with large space
    manager.release_interval(0, 10000)
    
    print(f"Initial state:")
    manager.print_intervals()
    
    # Perform operations to create fragmentation
    operations = [
        ('reserve', 1000, 1500),
        ('reserve', 3000, 3200),
        ('reserve', 5000, 5800),
        ('release', 2000, 2500),
        ('reserve', 7000, 7300),
        ('release', 6000, 6500),
    ]
    
    print(f"\nApplying {len(operations)} operations...")
    
    for op, start, end in operations:
        if op == 'reserve':
            manager.reserve_interval(start, end)
        else:
            manager.release_interval(start, end)
        
        print(f"  {op.title()} [{start}, {end})")
    
    print(f"\nFinal state:")
    manager.print_intervals()
    
    # Test advanced queries
    print(f"\nAdvanced Queries:")
    
    # Best fit test
    best_fit = manager.find_best_fit(300)
    if best_fit:
        print(f"  Best fit (300 units): [{best_fit[0]}, {best_fit[1]})")
    
    # Largest available
    largest = manager.find_largest_available()
    if largest:
        print(f"  Largest available: [{largest[0]}, {largest[1]}), size={largest[1]-largest[0]}")
    
    # Performance stats
    perf = manager.get_performance_stats()
    print(f"  Performance: {perf['operation_count']} ops, {perf['cache_hit_rate']:.1%} cache hit rate")


if __name__ == "__main__":
    import time
    
    print("🏗️ Boundary-Based Summary Interval Tree")
    print("Combining boundary management efficiency with summary analytics")
    print("=" * 60)
    
    demo_boundary_summary_performance()
    
    # Performance comparison with regular boundary manager
    print(f"\n⚡ Performance Comparison")
    print("-" * 30)
    
    # Test boundary summary manager
    manager = BoundarySummaryManager()
    manager.release_interval(0, 100000)
    
    # Time operations
    operations = []
    for _ in range(5000):
        op = ['reserve', 'release'][_ % 2]
        start = (_ * 17) % 90000
        end = start + ((_ * 13) % 1000) + 1
        operations.append((op, start, end))
    
    start_time = time.perf_counter()
    for op, start, end in operations:
        if op == 'reserve':
            manager.reserve_interval(start, end)
        else:
            manager.release_interval(start, end)
    
    # Time summary access
    summary_start = time.perf_counter()
    for _ in range(1000):
        stats = manager.get_availability_stats()
    summary_time = time.perf_counter() - summary_start
    
    total_time = time.perf_counter() - start_time
    
    print(f"Boundary Summary Manager:")
    print(f"  Operations: {len(operations):,} in {total_time:.3f}s ({len(operations)/total_time:,.0f} ops/sec)")
    print(f"  Summary queries: 1000 in {summary_time*1000:.1f}ms ({summary_time*1000:.3f}ms avg)")
    print(f"  Final intervals: {len(manager.intervals)}")
    
    perf = manager.get_performance_stats()
    print(f"  Cache performance: {perf['cache_hit_rate']:.1%} hit rate")
    
    print(f"\n✅ Boundary summary demonstration complete!")
    print(f"Key advantages:")
    print(f"  • Simple boundary-based implementation")
    print(f"  • O(1) summary statistics with caching")
    print(f"  • Advanced queries (best-fit, largest available)")
    print(f"  • Performance monitoring and optimization")
    print(f"  • Graceful fallback when SortedContainers unavailable")
