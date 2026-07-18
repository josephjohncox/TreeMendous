"""
Basic interval tree implementations in pure Python.

This package provides reference implementations of various interval tree strategies,
focusing on readability and algorithm demonstration before optimization.
"""

from .avl import IntervalNode, IntervalTree
from .avl_earliest import EarliestIntervalNode, EarliestIntervalTree
from .base import (
    IntervalManagerProtocol,
    IntervalNodeBase,
    IntervalNodeProtocol,
    IntervalTreeBase,
)
from .boundary import IntervalManager
from .boundary_summary import BoundarySummary, BoundarySummaryManager
from .protocols import (
    AvailabilityStats,
    BackendConfiguration,
    CoreIntervalManagerProtocol,
    EnhancedIntervalManagerProtocol,
    ImplementationType,
    IntervalResult,
    PerformanceStats,
    PerformanceTier,
    PerformanceTrackingProtocol,
    RandomizedProtocol,
)
from .summary import SummaryIntervalNode, SummaryIntervalTree, TreeSummary
from .treap import IntervalTreap, TreapNode

__all__ = [
    # Base abstractions (legacy)
    "IntervalNodeBase",
    "IntervalNodeProtocol",
    "IntervalTreeBase",
    "IntervalManagerProtocol",
    # Unified protocols (new)
    "CoreIntervalManagerProtocol",
    "EnhancedIntervalManagerProtocol",
    "PerformanceTrackingProtocol",
    "RandomizedProtocol",
    "IntervalResult",
    "AvailabilityStats",
    "PerformanceStats",
    "BackendConfiguration",
    "ImplementationType",
    "PerformanceTier",
    # AVL tree implementations
    "IntervalNode",
    "IntervalTree",
    "EarliestIntervalNode",
    "EarliestIntervalTree",
    # Boundary management
    "IntervalManager",
    "BoundarySummary",
    "BoundarySummaryManager",
    # Summary statistics tree
    "TreeSummary",
    "SummaryIntervalNode",
    "SummaryIntervalTree",
    # Randomized treap
    "TreapNode",
    "IntervalTreap",
]
