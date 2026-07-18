"""Compatibility wrappers for optional C++ extension modules."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

from ..basic.protocols import IntervalResult


def _optional_module(name: str) -> ModuleType | None:
    """Import an extension module, returning ``None`` when it is not built."""
    try:
        return import_module(f"{__name__}.{name}")
    except ImportError:
        return None


boundary = _optional_module("boundary")
treap = _optional_module("treap")
boundary_summary = _optional_module("boundary_summary")
summary = _optional_module("summary")

CPP_BOUNDARY_AVAILABLE = boundary is not None
CPP_TREAP_AVAILABLE = treap is not None
CPP_BOUNDARY_SUMMARY_AVAILABLE = boundary_summary is not None
CPP_SUMMARY_AVAILABLE = summary is not None


class ProtocolCompliantBoundarySummaryManager:
    """Normalize C++ boundary-summary query results to Python value objects."""

    def __init__(self) -> None:
        if boundary_summary is None:
            raise ImportError("C++ boundary_summary module not available")
        manager_type: Any = getattr(boundary_summary, "BoundarySummaryManager")
        self._manager: Any = manager_type()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)

    def find_best_fit(
        self, length: int, prefer_early: bool = True
    ) -> IntervalResult | None:
        """Find the best fit and normalize its result type."""
        result = self._manager.find_best_fit(length, prefer_early)
        if result is None:
            return None
        if isinstance(result, tuple):
            start, end = result
            return IntervalResult(start=start, end=end, length=length)
        return IntervalResult(start=result.start, end=result.end, length=result.length)

    def find_largest_available(self) -> IntervalResult | None:
        """Find the largest range and normalize its result type."""
        result = self._manager.find_largest_available()
        if result is None:
            return None
        if isinstance(result, tuple):
            start, end = result
            return IntervalResult(start=start, end=end, length=end - start)
        return IntervalResult(start=result.start, end=result.end, length=result.length)


BoundarySummaryManager: type[ProtocolCompliantBoundarySummaryManager] | None = (
    ProtocolCompliantBoundarySummaryManager if CPP_BOUNDARY_SUMMARY_AVAILABLE else None
)

__all__ = [
    "BoundarySummaryManager",
    "CPP_BOUNDARY_AVAILABLE",
    "CPP_BOUNDARY_SUMMARY_AVAILABLE",
    "CPP_SUMMARY_AVAILABLE",
    "CPP_TREAP_AVAILABLE",
    "ProtocolCompliantBoundarySummaryManager",
    "boundary",
    "boundary_summary",
    "summary",
    "treap",
]
