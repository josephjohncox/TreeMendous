from sortedcontainers import SortedDict

from treemendous.domain import (
    IntervalResult,
    Span,
    validate_coordinate,
    validate_length,
)


class IntervalManager:
    """Sorted geometry engine used by the boundary backend."""

    def __init__(self) -> None:
        self.intervals: SortedDict[int, tuple[int, None]] = SortedDict()
        self.total_available_length = 0

    def release_interval(self, start: int, end: int) -> None:
        Span(start, end)

        # Find position to insert or merge
        idx = self.intervals.bisect_left(start)
        # Check and merge with the previous interval if overlapping or adjacent.
        if idx > 0:
            prev_start = self.intervals.keys()[idx - 1]
            prev_end, _ = self.intervals[prev_start]
            if prev_end >= start:
                start = prev_start
                end = max(end, prev_end)
                idx -= 1
                del self.intervals[prev_start]
                self.total_available_length -= prev_end - prev_start

        # Merge with next intervals if overlapping
        while idx < len(self.intervals):
            curr_start = self.intervals.keys()[idx]
            curr_end, _ = self.intervals[curr_start]
            if curr_start > end:
                break
            end = max(end, curr_end)
            del self.intervals[curr_start]
            self.total_available_length -= curr_end - curr_start

        # Insert the new merged interval
        self.intervals[start] = (end, None)
        self.total_available_length += end - start

    def reserve_interval(self, start: int, end: int) -> None:
        Span(start, end)

        idx = self.intervals.bisect_left(start)

        if idx > 0:
            prev_start = self.intervals.keys()[idx - 1]
            prev_end, _ = self.intervals[prev_start]
            if prev_end > start:
                idx -= 1

        intervals_to_add: list[tuple[int, int]] = []
        keys_to_delete: list[int] = []

        while idx < len(self.intervals):
            curr_start = self.intervals.keys()[idx]
            curr_end, _ = self.intervals[curr_start]

            if curr_start >= end:
                break

            overlap_start = max(start, curr_start)
            overlap_end = min(end, curr_end)

            if overlap_start < overlap_end:
                # Mark current interval for removal
                keys_to_delete.append(curr_start)
                self.total_available_length -= curr_end - curr_start

                # Retain the non-overlapping geometry.
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
            self.intervals[s] = (e, None)
            self.total_available_length += e - s

    def find_interval(self, start: int, length: int) -> IntervalResult | None:
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

    def get_total_available_length(self) -> int:
        return self.total_available_length

    def print_intervals(self) -> None:
        print("Available intervals:")
        for start, (end, _) in self.intervals.items():
            print(f"[{start}, {end})")
        print(f"Total available length: {self.total_available_length}")

    def get_intervals(self) -> list[IntervalResult]:
        return [
            IntervalResult(start=start, end=end)
            for start, (end, _) in self.intervals.items()
        ]
