"""Dynamic lower-bound projections for fixed-dimensional box indexes."""

from __future__ import annotations

from bisect import bisect_left, insort
from dataclasses import dataclass
from typing import TYPE_CHECKING

from treemendous.multidimensional.domain import Box, BoxHandle

if TYPE_CHECKING:
    from collections.abc import Mapping

    from treemendous.multidimensional.domain import BoxEntry


_ProjectionRecord = tuple[int, int, BoxHandle]


@dataclass(frozen=True)
class _CartesianProjection:
    """Sequence-ordered Cartesian tree whose heap key is the lower bound."""

    records: tuple[_ProjectionRecord, ...]
    left: tuple[int, ...]
    right: tuple[int, ...]
    root: int

    @classmethod
    def build(
        cls,
        records: tuple[_ProjectionRecord, ...],
    ) -> _CartesianProjection:
        left = [-1] * len(records)
        right = [-1] * len(records)
        stack: list[int] = []
        for index, record in enumerate(records):
            previous = -1
            priority = (record[0], record[1])
            while stack:
                top_record = records[stack[-1]]
                if (top_record[0], top_record[1]) <= priority:
                    break
                previous = stack.pop()
            if stack:
                right[stack[-1]] = index
            left[index] = previous
            stack.append(index)
        root = stack[0] if stack else -1
        return cls(records, tuple(left), tuple(right), root)

    def prefix_handles(self, upper: int) -> tuple[BoxHandle, ...]:
        """Report ``lower < upper`` records directly in sequence order."""
        if self.root < 0:
            return ()
        handles: list[BoxHandle] = []
        # A nonnegative value visits a subtree. Its bitwise complement emits
        # that already-qualified node between its left and right subtrees.
        stack = [self.root]
        while stack:
            marker = stack.pop()
            if marker < 0:
                handles.append(self.records[~marker][2])
                continue
            record = self.records[marker]
            if record[0] >= upper:
                # Cartesian heap order proves that every descendant also has a
                # lower bound at least this large.
                continue
            right = self.right[marker]
            if right >= 0:
                stack.append(right)
            stack.append(~marker)
            left = self.left[marker]
            if left >= 0:
                stack.append(left)
        return tuple(handles)


@dataclass(frozen=True)
class _ProjectionState:
    projections: tuple[tuple[_ProjectionRecord, ...], ...]
    sequence_projections: tuple[_CartesianProjection, ...]


class AxisProjectionStrategy:
    """Copy-on-write sorted lower-bound projections for one fixed dimension."""

    algorithm = "axis_projection"

    def __init__(self, dimensions: int) -> None:
        empty = _CartesianProjection.build(())
        self._dimensions = dimensions
        self.initial_state = _ProjectionState(
            tuple(() for _ in range(dimensions)),
            tuple(empty for _ in range(dimensions)),
        )

    def _with_change(
        self,
        state: _ProjectionState,
        *,
        add: tuple[BoxHandle, Box] | None = None,
        remove: tuple[BoxHandle, Box] | None = None,
    ) -> _ProjectionState:
        replacement: list[tuple[_ProjectionRecord, ...]] = []
        sequence_replacement: list[_CartesianProjection] = []
        for axis, current in enumerate(state.projections):
            projection = list(current)
            if remove is not None:
                old_handle, old_box = remove
                old_record = (
                    old_box.lower[axis],
                    old_handle.sequence,
                    old_handle,
                )
                projection.remove(old_record)
            if add is not None:
                new_handle, new_box = add
                insort(
                    projection,
                    (new_box.lower[axis], new_handle.sequence, new_handle),
                    key=lambda record: (record[0], record[1]),
                )
            frozen_projection = tuple(projection)
            replacement.append(frozen_projection)
            sequence_records = tuple(
                sorted(frozen_projection, key=lambda record: record[1])
            )
            sequence_replacement.append(_CartesianProjection.build(sequence_records))
        return _ProjectionState(
            tuple(replacement),
            tuple(sequence_replacement),
        )

    def prepare_insert(
        self, state: _ProjectionState, handle: BoxHandle, box: Box
    ) -> _ProjectionState:
        return self._with_change(state, add=(handle, box))

    def prepare_update(
        self,
        state: _ProjectionState,
        handle: BoxHandle,
        old_box: Box,
        new_box: Box,
    ) -> _ProjectionState:
        if old_box == new_box:
            return state
        return self._with_change(state, add=(handle, new_box), remove=(handle, old_box))

    def prepare_remove(
        self, state: _ProjectionState, handle: BoxHandle, box: Box
    ) -> _ProjectionState:
        return self._with_change(state, remove=(handle, box))

    def candidate_handles(
        self,
        state: _ProjectionState,
        query: Box,
        entries: Mapping[BoxHandle, BoxEntry],
    ) -> tuple[BoxHandle, ...]:
        del entries
        selected_axis = 0
        selected_size: int | None = None
        for axis, projection in enumerate(state.projections):
            prefix_size = bisect_left(
                projection,
                (query.upper[axis], -1),
                key=lambda record: (record[0], record[1]),
            )
            if selected_size is None or prefix_size < selected_size:
                selected_axis = axis
                selected_size = prefix_size
        if selected_size == 0:
            return ()
        return state.sequence_projections[selected_axis].prefix_handles(
            query.upper[selected_axis]
        )

    def diagnostics(self, state: _ProjectionState) -> dict[str, object]:
        return {
            "projection_sizes": tuple(
                len(projection) for projection in state.projections
            )
        }
