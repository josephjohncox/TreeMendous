"""
Unified Protocol Definitions for Tree-Mendous

Defines consistent protocols and data structures that all implementations must follow.
This eliminates protocol drift and ensures interface consistency.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar, Protocol, Union, Dict, Any
from enum import Enum


# Type variables
D = TypeVar('D')  # Type variable for interval data
T = TypeVar('T', bound='IntervalNodeProtocol[D]')


class PerformanceTier(Enum):
    """Performance tier classification for implementations"""
    BASELINE = "baseline"
    OPTIMIZED = "optimized" 
    HIGH_PERFORMANCE = "high_performance"


class ImplementationType(Enum):
    """Type of interval tree implementation"""
    AVL_TREE = "avl_tree"
    BOUNDARY = "boundary"
    TREAP = "treap"
    SEGMENT_TREE = "segment_tree"
    SUMMARY_TREE = "summary_tree"


@dataclass(frozen=True)
class IntervalResult:
    """Standardized result for interval queries"""
    start: int
    end: int
    length: int = None
    data: Optional[Any] = None
    
    def __post_init__(self):
        if self.length is None:
            object.__setattr__(self, 'length', self.end - self.start)


@dataclass(frozen=True)
class AvailabilityStats:
    """Standardized availability statistics across all implementations"""
    total_free: int
    total_occupied: int = 0
    total_space: int = None
    free_chunks: int = 0
    largest_chunk: int = 0
    avg_chunk_size: float = 0.0
    utilization: float = 0.0
    fragmentation: float = 0.0
    free_density: float = 0.0
    bounds: Tuple[Optional[int], Optional[int]] = (None, None)
    
    def __post_init__(self):
        if self.total_space is None:
            object.__setattr__(self, 'total_space', self.total_free + self.total_occupied)
        
        if self.total_space > 0:
            object.__setattr__(self, 'utilization', self.total_occupied / self.total_space)
            object.__setattr__(self, 'free_density', self.total_free / self.total_space)
        
        if self.free_chunks > 0:
            object.__setattr__(self, 'avg_chunk_size', self.total_free / self.free_chunks)
        
        if self.total_free > 0 and self.largest_chunk > 0:
            object.__setattr__(self, 'fragmentation', 1.0 - (self.largest_chunk / self.total_free))


@dataclass(frozen=True)
class PerformanceStats:
    """Standardized performance statistics"""
    operation_count: int = 0
    cache_hits: int = 0
    cache_hit_rate: float = 0.0
    implementation_name: str = ""
    language: str = ""
    
    def __post_init__(self):
        if self.operation_count > 0:
            object.__setattr__(self, 'cache_hit_rate', self.cache_hits / self.operation_count)


@dataclass
class BackendConfiguration:
    """Configuration for a specific backend implementation"""
    implementation_id: str
    name: str
    language: str
    implementation_type: ImplementationType
    performance_tier: PerformanceTier
    features: List[str]
    available: bool = False
    estimated_speedup: float = 1.0
    constructor_args: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.constructor_args is None:
            self.constructor_args = {}


class IntervalNodeProtocol(Protocol[D]):
    """Protocol that all interval nodes must implement"""
    start: int
    end: int
    length: int
    height: int
    total_length: int
    data: Optional[D]
    left: Optional['IntervalNodeProtocol[D]']
    right: Optional['IntervalNodeProtocol[D]']
    
    def update_stats(self) -> None: ...
    def update_length(self) -> None: ...


class CoreIntervalManagerProtocol(Protocol[D]):
    """Core protocol that ALL interval managers must implement"""
    
    def release_interval(self, start: int, end: int, data: Optional[D] = None) -> None:
        """Add interval to available space"""
        ...
    
    def reserve_interval(self, start: int, end: int, data: Optional[D] = None) -> None:
        """Remove interval from available space"""
        ...
    
    def find_interval(self, start: int, length: int) -> Optional[IntervalResult]:
        """Find available interval of given length starting at or after start time"""
        ...
    
    def get_intervals(self) -> List[IntervalResult]:
        """Get all available intervals"""
        ...
    
    def get_total_available_length(self) -> int:
        """Get total available space"""
        ...


class EnhancedIntervalManagerProtocol(CoreIntervalManagerProtocol[D]):
    """Enhanced protocol for implementations with additional features"""
    
    def get_availability_stats(self) -> AvailabilityStats:
        """Get comprehensive availability statistics"""
        ...
    
    def find_best_fit(self, length: int, prefer_early: bool = True) -> Optional[IntervalResult]:
        """Find best-fit interval for given length"""
        ...
    
    def find_largest_available(self) -> Optional[IntervalResult]:
        """Find largest available interval"""
        ...


class PerformanceTrackingProtocol(Protocol):
    """Protocol for implementations that track performance metrics"""
    
    def get_performance_stats(self) -> PerformanceStats:
        """Get performance and caching statistics"""
        ...


class RandomizedProtocol(Protocol):
    """Protocol for randomized implementations (treaps, etc.)"""
    
    def sample_random_interval(self) -> Optional[IntervalResult]:
        """Sample a random available interval"""
        ...
    
    def verify_properties(self) -> bool:
        """Verify implementation-specific properties (e.g., treap properties)"""
        ...


# Primary protocol (all implementations must support this)
IntervalManagerProtocol = CoreIntervalManagerProtocol


def standardize_interval_result(result: Any) -> Optional[IntervalResult]:
    """Convert various result formats to standardized IntervalResult"""
    if result is None:
        return None
    
    if isinstance(result, IntervalResult):
        return result
    
    if isinstance(result, tuple):
        if len(result) == 2:
            start, end = result
            return IntervalResult(start=start, end=end)
        elif len(result) == 3:
            start, end, data = result
            return IntervalResult(start=start, end=end, data=data)
    
    if hasattr(result, 'start') and hasattr(result, 'end'):
        # Node object
        data = getattr(result, 'data', None)
        return IntervalResult(start=result.start, end=result.end, data=data)
    
    raise ValueError(f"Cannot standardize result: {type(result)}")


def standardize_intervals_list(intervals: Any) -> List[IntervalResult]:
    """Convert various interval list formats to standardized list"""
    if not intervals:
        return []
    
    result = []
    for interval in intervals:
        standardized = standardize_interval_result(interval)
        if standardized:
            result.append(standardized)
    
    return result


def standardize_availability_stats(stats: Any) -> AvailabilityStats:
    """Convert various stats formats to standardized AvailabilityStats"""
    if isinstance(stats, AvailabilityStats):
        return stats
    
    if isinstance(stats, dict):
        return AvailabilityStats(
            total_free=stats.get('total_free', 0),
            total_occupied=stats.get('total_occupied', 0),
            total_space=stats.get('total_space'),
            free_chunks=stats.get('free_chunks', 0),
            largest_chunk=stats.get('largest_chunk', 0),
            avg_chunk_size=stats.get('avg_chunk_size', 0.0),
            utilization=stats.get('utilization', 0.0),
            fragmentation=stats.get('fragmentation', 0.0),
            free_density=stats.get('free_density', 0.0),
            bounds=stats.get('bounds', (None, None))
        )
    
    # Handle C++ object attributes
    if hasattr(stats, 'total_free'):
        return AvailabilityStats(
            total_free=stats.total_free,
            total_occupied=getattr(stats, 'total_occupied', 0),
            total_space=getattr(stats, 'total_space', None),
            free_chunks=getattr(stats, 'free_chunks', 0),
            largest_chunk=getattr(stats, 'largest_chunk', 0),
            avg_chunk_size=getattr(stats, 'avg_chunk_size', 0.0),
            utilization=getattr(stats, 'utilization', 0.0),
            fragmentation=getattr(stats, 'fragmentation', 0.0),
            free_density=getattr(stats, 'free_density', 0.0),
            bounds=getattr(stats, 'bounds', (None, None))
        )
    
    raise ValueError(f"Cannot standardize stats: {type(stats)}")


def standardize_performance_stats(stats: Any) -> PerformanceStats:
    """Convert various performance stats formats to standardized PerformanceStats"""
    if isinstance(stats, PerformanceStats):
        return stats
    
    if isinstance(stats, dict):
        return PerformanceStats(
            operation_count=stats.get('operation_count', 0),
            cache_hits=stats.get('cache_hits', 0),
            cache_hit_rate=stats.get('cache_hit_rate', 0.0),
            implementation_name=stats.get('implementation', ''),
            language=stats.get('language', '')
        )
    
    # Handle C++ object attributes
    if hasattr(stats, 'operation_count'):
        return PerformanceStats(
            operation_count=stats.operation_count,
            cache_hits=getattr(stats, 'cache_hits', 0),
            cache_hit_rate=getattr(stats, 'cache_hit_rate', 0.0),
            implementation_name=getattr(stats, 'implementation', ''),
            language=getattr(stats, 'language', '')
        )
    
    raise ValueError(f"Cannot standardize performance stats: {type(stats)}")
