from sortedcontainers import SortedDict

from treemendous.domain import (
    IntervalResult,
    MutationResult,
    Span,
    validate_coordinate,
    validate_length,
)


class IntervalManager:
    """Sorted geometry engine used by the boundary backend."""

    _treemendous_authoritative_geometry = True

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

    def release_with_delta(self, start: int, end: int) -> MutationResult:
        Span(start, end)
        requested_end = end
        merged_start = start
        merged_end = end
        changed: list[Span] = []
        keys_to_delete: list[int] = []
        removed_measure = 0
        cursor = start
        index = self.intervals.bisect_left(start)

        if index > 0:
            previous_start = self.intervals.keys()[index - 1]
            previous_end, _ = self.intervals[previous_start]
            if previous_end >= start:
                if end <= previous_end:
                    return MutationResult((), 0, True)
                if previous_end > cursor:
                    cursor = min(previous_end, requested_end)
                merged_start = previous_start
                merged_end = max(merged_end, previous_end)
                keys_to_delete.append(previous_start)
                removed_measure += previous_end - previous_start

        while index < len(self.intervals):
            current_start = self.intervals.keys()[index]
            current_end, _ = self.intervals[current_start]
            if current_start > merged_end:
                break
            if current_end > cursor and current_start < requested_end:
                if current_start > cursor:
                    changed.append(Span(cursor, min(current_start, requested_end)))
                cursor = max(cursor, min(current_end, requested_end))
            merged_end = max(merged_end, current_end)
            keys_to_delete.append(current_start)
            removed_measure += current_end - current_start
            index += 1

        if cursor < requested_end:
            changed.append(Span(cursor, requested_end))
        changed_tuple = tuple(changed)
        result = MutationResult(
            changed_tuple,
            sum(span.length for span in changed_tuple),
            not changed_tuple,
        )
        prospective_total = (
            self.total_available_length - removed_measure + merged_end - merged_start
        )
        for key in keys_to_delete:
            del self.intervals[key]
        self.intervals[merged_start] = (merged_end, None)
        self.total_available_length = prospective_total
        return result

    def reserve_with_delta(
        self, start: int, end: int, require_covered: bool
    ) -> MutationResult:
        target = Span(start, end)
        index = self.intervals.bisect_left(start)
        if index > 0:
            previous_start = self.intervals.keys()[index - 1]
            previous_end, _ = self.intervals[previous_start]
            if previous_end > start:
                index -= 1

        changed: list[Span] = []
        intervals_to_add: list[tuple[int, int]] = []
        keys_to_delete: list[int] = []
        removed_measure = 0
        while index < len(self.intervals):
            current_start = self.intervals.keys()[index]
            current_end, _ = self.intervals[current_start]
            if current_start >= end:
                break
            overlap_start = max(start, current_start)
            overlap_end = min(end, current_end)
            if overlap_start < overlap_end:
                changed.append(Span(overlap_start, overlap_end))
                removed_measure += current_end - current_start
                keys_to_delete.append(current_start)
                if current_start < start:
                    intervals_to_add.append((current_start, start))
                if current_end > end:
                    intervals_to_add.append((end, current_end))
            index += 1

        changed_tuple = tuple(changed)
        changed_length = sum(span.length for span in changed_tuple)
        covered = changed_length == target.length
        if require_covered and not covered:
            return MutationResult((), 0, False)
        result = MutationResult(changed_tuple, changed_length, covered)

        for key in keys_to_delete:
            del self.intervals[key]
        self.total_available_length -= removed_measure
        for remaining_start, remaining_end in intervals_to_add:
            self.intervals[remaining_start] = (remaining_end, None)
            self.total_available_length += remaining_end - remaining_start
        return result

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

    def allocate_interval(
        self, start: int, length: int, not_after: int | None
    ) -> IntervalResult | None:
        result = self.find_interval(start, length)
        if result is None or (not_after is not None and result.end > not_after):
            return None
        self.reserve_interval(result.start, result.end)
        return result

    def find_overlapping_intervals(self, start: int, end: int) -> list[IntervalResult]:
        Span(start, end)
        result: list[IntervalResult] = []
        index = max(0, self.intervals.bisect_right(start) - 1)
        while index < len(self.intervals):
            interval_start = self.intervals.keys()[index]
            interval_end, _ = self.intervals[interval_start]
            if interval_start >= end:
                break
            if interval_end > start:
                result.append(IntervalResult(interval_start, interval_end))
            index += 1
        return result

    def get_total_available_length(self) -> int:
        return self.total_available_length

    def get_interval_count(self) -> int:
        return len(self.intervals)

    def get_largest_available_length(self) -> int:
        return max(
            (end - start for start, (end, _) in self.intervals.items()),
            default=0,
        )

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
