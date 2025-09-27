"""
Basic interval tree implementations in pure Python.

This package provides reference implementations of various interval tree strategies,
focusing on readability and algorithm demonstration before optimization.
"""

from .base import IntervalNodeBase, IntervalNodeProtocol, IntervalTreeBase, IntervalManagerProtocol
from .avl import IntervalNode, IntervalTree
from .avl_earliest import EarliestIntervalNode, EarliestIntervalTree
from .boundary import IntervalManager
from .segment import SegmentTreeNode, SegmentTree
from .summary import TreeSummary, SummaryIntervalNode, SummaryIntervalTree
from .treap import TreapNode, IntervalTreap

__all__ = [
    # Base abstractions
    'IntervalNodeBase',
    'IntervalNodeProtocol', 
    'IntervalTreeBase',
    'IntervalManagerProtocol',
    
    # AVL tree implementations
    'IntervalNode',
    'IntervalTree',
    'EarliestIntervalNode', 
    'EarliestIntervalTree',
    
    # Boundary management
    'IntervalManager',
    
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
