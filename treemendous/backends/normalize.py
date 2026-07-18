"""Normalize heterogeneous raw backend results at the adapter seam."""

from __future__ import annotations

from typing import Any

from treemendous.domain import IntervalResult


def normalize_interval(result: Any) -> IntervalResult | None:
    if result is None:
        return None
    if isinstance(result, IntervalResult):
        return result
    if isinstance(result, tuple):
        if len(result) == 2:
            return IntervalResult(result[0], result[1])
        if len(result) == 3:
            return IntervalResult(result[0], result[1], data=result[2])
    if hasattr(result, "start") and hasattr(result, "end"):
        return IntervalResult(
            result.start,
            result.end,
            data=getattr(result, "data", None),
        )
    raise ValueError(f"cannot normalize interval result: {type(result)}")


def normalize_intervals(intervals: Any) -> list[IntervalResult]:
    return [
        interval
        for raw in (intervals or ())
        if (interval := normalize_interval(raw)) is not None
    ]
