"""
Tree-Mendous: High-Performance Interval Tree Implementations

A comprehensive collection of interval tree implementations with unified backend management
and consistent protocols across Python and C++ implementations.
"""

# Export unified backend system
from treemendous.backend_manager import (
    TreeMendousBackendManager,
    UnifiedIntervalManager,
    create_interval_tree,
    list_available_backends,
    print_backend_status,
    get_backend_manager,
)

# Export protocol definitions
from treemendous.basic.protocols import (
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

# Export core implementations for direct use
from treemendous.basic.summary import SummaryIntervalTree
from treemendous.basic.treap import IntervalTreap
from treemendous.basic.boundary_summary import BoundarySummaryManager

__version__ = "0.2.4"
__author__ = "Joseph Cox"
__description__ = "High-performance interval tree implementations with unified backend management"

# Backwards compatibility
def create_summary_tree():
    """Create summary tree (backwards compatibility)"""
    return create_interval_tree("py_summary")

def create_treap(random_seed: int = 42):
    """Create treap (backwards compatibility)"""  
    return create_interval_tree("py_treap")

__all__ = [
    # Unified backend system (primary interface)
    "create_interval_tree",
    "TreeMendousBackendManager", 
    "UnifiedIntervalManager",
    "list_available_backends",
    "print_backend_status",
    "get_backend_manager",
    
    # Protocol definitions
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
    
    # Direct implementation access
    "SummaryIntervalTree",
    "IntervalTreap", 
    "BoundarySummaryManager",
    
    # Backwards compatibility
    "create_summary_tree",
    "create_treap",
]
