from sortedcontainers import SortedDict
from typing import List, Optional, Tuple
from .protocols import CoreIntervalManagerProtocol, IntervalResult, AvailabilityStats

class IntervalManager(CoreIntervalManagerProtocol[None]):
    def __init__(self) -> None:
        # Intervals are stored as {start: end}
        self.intervals: SortedDict[int, int] = SortedDict()
        self.total_available_length: int = 0

    def release_interval(self, start: int, end: int) -> None:
        if start >= end:
            return

        # Find position to insert or merge
        idx = self.intervals.bisect_left(start)

        # Check and merge with previous interval if overlapping or adjacent
        if idx > 0:
            prev_start = self.intervals.keys()[idx - 1]
            prev_end = self.intervals[prev_start]
            if prev_end >= start:
                start = prev_start
                end = max(end, prev_end)
                idx -= 1
                del self.intervals[prev_start]
                self.total_available_length -= prev_end - prev_start

        # Merge with next intervals if overlapping
        while idx < len(self.intervals):
            curr_start = self.intervals.keys()[idx]
            curr_end = self.intervals[curr_start]
            if curr_start > end:
                break
            end = max(end, curr_end)
            del self.intervals[curr_start]
            self.total_available_length -= curr_end - curr_start

        # Insert the new merged interval
        self.intervals[start] = end
        self.total_available_length += end - start

    def reserve_interval(self, start: int, end: int) -> None:
        if start >= end:
            return

        idx = self.intervals.bisect_left(start)

        if idx > 0:
            prev_start = self.intervals.keys()[idx - 1]
            prev_end = self.intervals[prev_start]
            if prev_end > start:
                idx -= 1

        intervals_to_add: List[Tuple[int, int]] = []
        keys_to_delete: List[int] = []

        while idx < len(self.intervals):
            curr_start = self.intervals.keys()[idx]
            curr_end = self.intervals[curr_start]

            if curr_start >= end:
                break

            overlap_start = max(start, curr_start)
            overlap_end = min(end, curr_end)

            if overlap_start < overlap_end:
                # Mark current interval for removal
                keys_to_delete.append(curr_start)
                self.total_available_length -= curr_end - curr_start

                # Add non-overlapping intervals
                if curr_start < start:
                    intervals_to_add.append((curr_start, start))
                if curr_end > end:
                    intervals_to_add.append((end, curr_end))

            idx += 1

        # Remove intervals after iteration
        for key in keys_to_delete:
            del self.intervals[key]

        # Add new intervals
        for s, e in intervals_to_add:
            self.intervals[s] = e
            self.total_available_length += e - s

    def find_interval(self, start: int, length: int) -> Optional[IntervalResult]:
        idx = self.intervals.bisect_left(start)
        intervals_keys = self.intervals.keys()

        # Check the interval at idx
        if idx < len(intervals_keys):
            s = intervals_keys[idx]
            e = self.intervals[s]
            if s <= start < e and e - start >= length:
                return IntervalResult(start=start, end=start + length)
            elif s > start and e - s >= length:
                return IntervalResult(start=s, end=s + length)

        # Check the previous interval
        if idx > 0:
            idx -= 1
            s = intervals_keys[idx]
            e = self.intervals[s]
            if s <= start < e and e - start >= length:
                return IntervalResult(start=start, end=start + length)
            elif start < s and e - s >= length:
                return IntervalResult(start=s, end=s + length)

        return None

    def get_total_available_length(self) -> int:
        return self.total_available_length

    def print_intervals(self) -> None:
        print("Available intervals:")
        for s, e in self.intervals.items():
            print(f"[{s}, {e})")
        print(f"Total available length: {self.total_available_length}")
    
    def get_intervals(self) -> List[IntervalResult]:
        return [IntervalResult(start=start, end=end) for start, end in self.intervals.items()]

# Example usage:
if __name__ == "__main__":
    manager = IntervalManager()
    # Initially, the whole interval [0, 100) is available
    manager.release_interval(0, 100)
    print("Initial intervals:")
    manager.print_intervals()

    # Schedule interval [10, 20)
    manager.reserve_interval(10, 20)
    print("\nAfter scheduling [10, 20):")
    manager.print_intervals()

    # Schedule interval [15, 25)
    manager.reserve_interval(15, 25)
    print("\nAfter scheduling [15, 25):")
    manager.print_intervals()

    # Unschedule interval [10, 20)
    manager.release_interval(10, 20)
    print("\nAfter unscheduling [10, 20):")
    manager.print_intervals()

    # Schedule two adjacent intervals [30, 40) and [40, 50)
    manager.reserve_interval(30, 40)
    manager.reserve_interval(40, 50)
    print("\nAfter scheduling two adjacent intervals [30, 40) and [40, 50):")
    manager.print_intervals()

    # Release overlapping interval [35, 45)
    manager.release_interval(35, 45)
    print("\nAfter releasing overlapping interval [35, 45):")
    manager.print_intervals()

    # Test finding intervals
    result = manager.find_interval(0, 15)
    print(f"\nFinding interval of length 15 starting from 0: {result}")

    # Test complete overlap
    manager.reserve_interval(60, 80)
    print("\nAfter reserving [60, 80):")
    manager.print_intervals()
    
    manager.release_interval(65, 75)
    print("\nAfter releasing internal interval [65, 75):")
    manager.print_intervals()