"""Canonical Tree-Mendous range-set interface."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version

from treemendous.backends.registry import BackendRegistry, create_range_set
from treemendous.backends.types import (
    BackendDecision,
    BackendRequest,
    BackendSpec,
    Capability,
)
from treemendous.domain import (
    AvailabilityStats,
    BackendInvalidError,
    BackendUnavailableError,
    DomainInput,
    IntervalResult,
    ManagedDomain,
    ManagedDomainRequiredError,
    MutationResult,
    RangeSnapshot,
    Span,
)
from treemendous.policies import (
    JoinPayloadPolicy,
    OrderedPayloadPolicy,
    PayloadPolicy,
    UniformPayloadPolicy,
)
from treemendous.protocols import RangeSetProtocol
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
    "Exact integer range sets with Python/C++ backends, atomic native batches, "
    "experimental multidimensional indexes, and 50 application engines."
)


__all__ = [
    "AvailabilityStats",
    "BackendDecision",
    "BackendInvalidError",
    "BackendRegistry",
    "BackendRequest",
    "BackendSpec",
    "BackendUnavailableError",
    "Capability",
    "DomainInput",
    "IntervalResult",
    "JoinPayloadPolicy",
    "ManagedDomain",
    "ManagedDomainRequiredError",
    "MutationResult",
    "OrderedPayloadPolicy",
    "PayloadPolicy",
    "RangeSet",
    "RangeSetProtocol",
    "RangeSnapshot",
    "Span",
    "UniformPayloadPolicy",
    "create_range_set",
]
