"""
C++ Implementation Wrappers

This module provides Python wrappers for C++ implementations that ensure
protocol conformance by converting return types to standardized formats.
"""

from typing import Optional
from ..basic.protocols import IntervalResult

# Import C++ modules with fallback handling
try:
    from . import boundary
    CPP_BOUNDARY_AVAILABLE = True
except ImportError:
    CPP_BOUNDARY_AVAILABLE = False

try:
    from . import treap
    CPP_TREAP_AVAILABLE = True
except ImportError:
    CPP_TREAP_AVAILABLE = False

try:
    from . import boundary_summary
    CPP_BOUNDARY_SUMMARY_AVAILABLE = True
except ImportError:
    CPP_BOUNDARY_SUMMARY_AVAILABLE = False

try:
    from . import summary
    CPP_SUMMARY_AVAILABLE = True
except ImportError:
    CPP_SUMMARY_AVAILABLE = False


class ProtocolCompliantBoundarySummaryManager:
    """Wrapper for C++ BoundarySummaryManager that ensures protocol compliance"""
    
    def __init__(self):
        if not CPP_BOUNDARY_SUMMARY_AVAILABLE:
            raise ImportError("C++ boundary_summary module not available")
        self._manager = boundary_summary.BoundarySummaryManager()
    
    def __getattr__(self, name):
        # Delegate all other methods to the underlying manager
        return getattr(self._manager, name)
    
    def find_best_fit(self, length: int, prefer_early: bool = True) -> Optional[IntervalResult]:
        """Find best-fit interval with IntervalResult return type"""
        result = self._manager.find_best_fit(length, prefer_early)
        if result is None:
            return None
        
        # Convert tuple result to IntervalResult
        if isinstance(result, tuple):
            start, end = result
            return IntervalResult(start=start, end=end, length=length)
        else:
            # Already an IntervalResult-like object
            return IntervalResult(start=result.start, end=result.end, length=result.length)
    
    def find_largest_available(self) -> Optional[IntervalResult]:
        """Find largest available interval with IntervalResult return type"""
        result = self._manager.find_largest_available()
        if result is None:
            return None
        
        # Convert tuple result to IntervalResult  
        if isinstance(result, tuple):
            start, end = result
            return IntervalResult(start=start, end=end, length=end - start)
        else:
            # Already an IntervalResult-like object
            return IntervalResult(start=result.start, end=result.end, length=result.length)


# Export the protocol-compliant wrapper
if CPP_BOUNDARY_SUMMARY_AVAILABLE:
    BoundarySummaryManager = ProtocolCompliantBoundarySummaryManager
else:
    BoundarySummaryManager = None
