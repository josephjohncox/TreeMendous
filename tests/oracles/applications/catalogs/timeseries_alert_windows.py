"""Naive alert/suppression priority oracle."""

from __future__ import annotations


def active(
    rows: list[tuple[str, str, str, int, int, int]], series: str, time: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return firing and suppressed alert IDs by a bounded scan."""
    current = [row for row in rows if row[1] == series and row[4] <= time < row[5]]
    threshold = max(
        (row[3] for row in current if row[2] == "suppression"), default=None
    )
    alerts = [row for row in current if row[2] == "alert"]
    firing = tuple(row[0] for row in alerts if threshold is None or row[3] > threshold)
    suppressed = tuple(
        row[0] for row in alerts if threshold is not None and row[3] <= threshold
    )
    return firing, suppressed
