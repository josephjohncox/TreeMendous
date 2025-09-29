"""
Basic interval tree implementations in pure Python.

This package provides reference implementations of various interval tree strategies,
focusing on readability and algorithm demonstration before optimization.
"""

from .base import IntervalNodeBase, IntervalNodeProtocol, IntervalTreeBase, IntervalManagerProtocol
from .protocols import (
    CoreIntervalManagerProtocol,
    EnhancedIntervalManagerProtocol,
    PerformanceTrackingProtocol,
    RandomizedProtocol,
    IntervalResult,
    AvailabilityStats,
    PerformanceStats,
    BackendConfiguration,
    ImplementationType,
    PerformanceTier,
)
from .avl import IntervalNode, IntervalTree
from .avl_earliest import EarliestIntervalNode, EarliestIntervalTree
from .boundary import IntervalManager
from .segment import SegmentTreeNode, SegmentTree
from .summary import TreeSummary, SummaryIntervalNode, SummaryIntervalTree
from .treap import TreapNode, IntervalTreap
from .boundary_summary import BoundarySummary, BoundarySummaryManager

__all__ = [
    # Base abstractions (legacy)
    'IntervalNodeBase',
    'IntervalNodeProtocol', 
    'IntervalTreeBase',
    'IntervalManagerProtocol',
    
    # Unified protocols (new)
    'CoreIntervalManagerProtocol',
    'EnhancedIntervalManagerProtocol',
    'PerformanceTrackingProtocol',
    'RandomizedProtocol',
    'IntervalResult',
    'AvailabilityStats',
    'PerformanceStats',
    'BackendConfiguration',
    'ImplementationType',
    'PerformanceTier',
    
    # AVL tree implementations
    'IntervalNode',
    'IntervalTree',
    'EarliestIntervalNode', 
    'EarliestIntervalTree',
    
    # Boundary management
    'IntervalManager',
    'BoundarySummary',
    'BoundarySummaryManager',
    
    # Segment tree
    'SegmentTreeNode',
    'SegmentTree', 
    
    # Summary statistics tree
    'TreeSummary',
    'SummaryIntervalNode',
    'SummaryIntervalTree',
    
    # Randomized treap
    'TreapNode',
    'IntervalTreap',
]
