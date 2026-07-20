"""Naive distributed trace overlap and ancestry oracle."""

from __future__ import annotations

TraceRow = tuple[str, str, str | None, int, int]


def overlapping(
    rows: list[TraceRow],
    trace_id: str,
    start: int,
    end: int,
) -> tuple[str, ...]:
    """Return overlapping span IDs by linear scan."""
    return tuple(
        span_id
        for row_trace, span_id, _parent, left, right in rows
        if row_trace == trace_id and left < end and start < right
    )


def critical_path(rows: list[TraceRow], trace_id: str) -> tuple[str, ...]:
    """Enumerate root-to-leaf chains and select summed-duration maximum."""
    records = [row for row in rows if row[0] == trace_id]
    by_id = {row[1]: row for row in records}
    insertion_order = {row[1]: index for index, row in enumerate(records)}

    for row in records:
        seen: set[str] = set()
        cursor = row
        while True:
            span_id = cursor[1]
            if span_id in seen:
                raise ValueError("trace ancestry contains a cycle")
            seen.add(span_id)
            parent_id = cursor[2]
            if parent_id is None or parent_id not in by_id:
                break
            cursor = by_id[parent_id]

    parent_ids = {row[2] for row in records if row[2] is not None and row[2] in by_id}
    leaves = [row for row in records if row[1] not in parent_ids]
    choices: list[tuple[int, tuple[int, ...], tuple[str, ...]]] = []
    for leaf in leaves:
        reversed_chain = [leaf]
        cursor = leaf
        while cursor[2] is not None and cursor[2] in by_id:
            cursor = by_id[cursor[2]]
            reversed_chain.append(cursor)
        chain = tuple(reversed(reversed_chain))
        score = sum(row[4] - row[3] for row in chain)
        order_key = tuple(-insertion_order[row[1]] for row in chain)
        choices.append((score, order_key, tuple(row[1] for row in chain)))
    if not choices:
        return ()
    return max(choices, key=lambda choice: (choice[0], choice[1]))[2]
