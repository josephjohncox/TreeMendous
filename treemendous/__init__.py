"""Tree-Mendous public range-set interface."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version

from treemendous.backend_manager import (
    TreeMendousBackendManager,
    UnifiedIntervalManager,
    create_interval_tree,
    create_range_set,
    get_backend_manager,
    list_available_backends,
    print_backend_status,
)
from treemendous.backends import (
    BackendDecision,
    BackendRequest,
    BackendSpec,
    Capability,
)
from treemendous.basic.boundary_summary import BoundarySummaryManager
from treemendous.basic.protocols import (
    BackendConfiguration,
    CoreIntervalManagerProtocol,
    EnhancedIntervalManagerProtocol,
    ImplementationType,
    PerformanceStats,
    PerformanceTier,
    PerformanceTrackingProtocol,
    RandomizedProtocol,
)
from treemendous.basic.summary import SummaryIntervalTree
from treemendous.basic.treap import IntervalTreap
from treemendous.domain import (
    AvailabilityStats,
    BackendInvalidError,
    BackendUnavailableError,
    IntervalResult,
    ManagedDomain,
    ManagedDomainRequiredError,
    MutationResult,
    RangeSnapshot,
    Span,
    UnsupportedCapabilityError,
)
from treemendous.policies import (
    JoinPayloadPolicy,
    OrderedPayloadPolicy,
    UniformPayloadPolicy,
)
from treemendous.rangeset import RangeSet


def _resolve_version() -> str:
    """Return installed metadata version or a clear source-checkout fallback."""
    try:
        return _metadata_version("treemendous")
    except PackageNotFoundError:
        return "0.0.0+dev"


__version__ = _resolve_version()
__author__ = "Joseph Cox"
__description__ = (
    "High-performance interval tree implementations with unified backend management"
)


def create_summary_tree():
    """Create the legacy Python summary manager."""
    return create_interval_tree("py_summary")


def create_treap(random_seed: int = 42):
    """Create the legacy Python treap manager with the requested seed."""
    return create_interval_tree("py_treap", random_seed=random_seed)


__all__ = [
    "AvailabilityStats",
    "BackendConfiguration",
    "BackendDecision",
    "BackendInvalidError",
    "BackendRequest",
    "BackendSpec",
    "BackendUnavailableError",
    "BoundarySummaryManager",
    "Capability",
    "CoreIntervalManagerProtocol",
    "EnhancedIntervalManagerProtocol",
    "ImplementationType",
    "IntervalResult",
    "IntervalTreap",
    "JoinPayloadPolicy",
    "ManagedDomain",
    "ManagedDomainRequiredError",
    "MutationResult",
    "OrderedPayloadPolicy",
    "PerformanceStats",
    "PerformanceTier",
    "PerformanceTrackingProtocol",
    "RandomizedProtocol",
    "RangeSet",
    "RangeSnapshot",
    "Span",
    "SummaryIntervalTree",
    "TreeMendousBackendManager",
    "UnifiedIntervalManager",
    "UniformPayloadPolicy",
    "UnsupportedCapabilityError",
    "create_interval_tree",
    "create_range_set",
    "create_summary_tree",
    "create_treap",
    "get_backend_manager",
    "list_available_backends",
    "print_backend_status",
]
