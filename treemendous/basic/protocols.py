"""Compatibility protocols and canonical public value re-exports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, TypeVar

from treemendous.domain import AvailabilityStats, IntervalResult

D = TypeVar("D")
D_contra = TypeVar("D_contra", contravariant=True)


class PerformanceTier(Enum):
    BASELINE = "baseline"
    OPTIMIZED = "optimized"
    HIGH_PERFORMANCE = "high_performance"


class ImplementationType(Enum):
    AVL_TREE = "avl_tree"
    BOUNDARY = "boundary"
    TREAP = "treap"
    SEGMENT_TREE = "segment_tree"
    SUMMARY_TREE = "summary_tree"


@dataclass(frozen=True)
class PerformanceStats:
    operation_count: int = 0
    cache_hits: int = 0
    cache_hit_rate: float = 0.0
    implementation_name: str = ""
    language: str = ""

    def __post_init__(self) -> None:
        if self.operation_count > 0:
            object.__setattr__(
                self, "cache_hit_rate", self.cache_hits / self.operation_count
            )


@dataclass(frozen=True)
class BackendConfiguration:
    """Deprecated immutable compatibility view of a backend descriptor."""

    implementation_id: str
    name: str
    language: str
    implementation_type: ImplementationType
    performance_tier: PerformanceTier
    features: tuple[str, ...] | list[str]
    available: bool = False
    estimated_speedup: float = 1.0
    constructor_args: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(
            self, "constructor_args", MappingProxyType(dict(self.constructor_args))
        )


class IntervalNodeProtocol(Protocol[D]):
    start: int
    end: int
    length: int
    height: int
    total_length: int
    data: D | None
    left: IntervalNodeProtocol[D] | None
    right: IntervalNodeProtocol[D] | None

    def update_stats(self) -> None: ...
    def update_length(self) -> None: ...


class CoreIntervalManagerProtocol(Protocol[D_contra]):
    def release_interval(
        self, start: int, end: int, data: D_contra | None = None
    ) -> None: ...
    def reserve_interval(
        self, start: int, end: int, data: D_contra | None = None
    ) -> None: ...
    def find_interval(self, start: int, length: int) -> IntervalResult | None: ...
    def get_intervals(self) -> list[IntervalResult]: ...
    def get_total_available_length(self) -> int: ...


class EnhancedIntervalManagerProtocol(
    CoreIntervalManagerProtocol[D_contra], Protocol[D_contra]
):
    def get_availability_stats(self) -> AvailabilityStats | None: ...
    def find_best_fit(
        self, length: int, prefer_early: bool = True
    ) -> IntervalResult | None: ...
    def find_largest_available(self) -> IntervalResult | None: ...


class PerformanceTrackingProtocol(Protocol):
    def get_performance_stats(self) -> PerformanceStats: ...


class RandomizedProtocol(Protocol):
    def sample_random_interval(self) -> IntervalResult | None: ...
    def verify_properties(self) -> bool: ...


IntervalManagerProtocol = CoreIntervalManagerProtocol


def standardize_interval_result(result: Any) -> IntervalResult | None:
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
            result.start, result.end, data=getattr(result, "data", None)
        )
    raise ValueError(f"Cannot standardize result: {type(result)}")


def standardize_intervals_list(intervals: Any) -> list[IntervalResult]:
    return [
        item
        for raw in (intervals or ())
        if (item := standardize_interval_result(raw)) is not None
    ]


def standardize_availability_stats(stats: Any) -> AvailabilityStats:
    if isinstance(stats, AvailabilityStats):
        return stats
    if isinstance(stats, dict):
        return AvailabilityStats(
            total_free=stats.get("total_free", 0),
            total_occupied=stats.get("total_occupied", 0),
            total_space=stats.get("total_space"),
            free_chunks=stats.get("free_chunks", 0),
            largest_chunk=stats.get("largest_chunk", 0),
            bounds=stats.get("bounds", (None, None)),
        )
    if hasattr(stats, "total_free"):
        return AvailabilityStats(
            total_free=stats.total_free,
            total_occupied=getattr(stats, "total_occupied", 0),
            total_space=getattr(stats, "total_space", None),
            free_chunks=getattr(stats, "free_chunks", 0),
            largest_chunk=getattr(stats, "largest_chunk", 0),
            bounds=getattr(stats, "bounds", (None, None)),
        )
    raise ValueError(f"Cannot standardize stats: {type(stats)}")


def standardize_performance_stats(stats: Any) -> PerformanceStats:
    if isinstance(stats, PerformanceStats):
        return stats
    if isinstance(stats, dict):
        return PerformanceStats(
            operation_count=stats.get("operation_count", 0),
            cache_hits=stats.get("cache_hits", 0),
            implementation_name=stats.get("implementation", ""),
            language=stats.get("language", ""),
        )
    if hasattr(stats, "operation_count"):
        return PerformanceStats(
            operation_count=stats.operation_count,
            cache_hits=getattr(stats, "cache_hits", 0),
            implementation_name=getattr(stats, "implementation", ""),
            language=getattr(stats, "language", ""),
        )
    raise ValueError(f"Cannot standardize performance stats: {type(stats)}")


__all__ = [
    "AvailabilityStats",
    "BackendConfiguration",
    "CoreIntervalManagerProtocol",
    "EnhancedIntervalManagerProtocol",
    "ImplementationType",
    "IntervalManagerProtocol",
    "IntervalNodeProtocol",
    "IntervalResult",
    "PerformanceStats",
    "PerformanceTier",
    "PerformanceTrackingProtocol",
    "RandomizedProtocol",
    "standardize_availability_stats",
    "standardize_interval_result",
    "standardize_intervals_list",
    "standardize_performance_stats",
]
