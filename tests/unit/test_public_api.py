"""The 1.0 public surface has no compatibility aliases or raw backends."""

from __future__ import annotations

from importlib.util import find_spec

import treemendous

PUBLIC_API = {
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
}

REMOVED_MODULES = (
    "treemendous.backend_manager",
    "treemendous.basic.protocols",
    "treemendous.basic.segment",
    "treemendous.backends.adapters.cpp",
    "treemendous.backends.adapters.cuda",
    "treemendous.backends.adapters.metal",
    "treemendous.backends.adapters.python",
    "treemendous.cpp.boundary_optimized",
    "treemendous.cpp.metal.mixed",
    "treemendous.cpp.summary",
)


def test_top_level_exports_are_exactly_the_canonical_api() -> None:
    assert set(treemendous.__all__) == PUBLIC_API


def test_removed_modules_do_not_exist() -> None:
    for module_name in REMOVED_MODULES:
        assert find_spec(module_name) is None, module_name


def test_internal_backend_packages_export_nothing() -> None:
    import treemendous.basic as basic
    import treemendous.cpp as cpp
    import treemendous.cpp.gpu as gpu
    import treemendous.cpp.metal as metal

    assert basic.__all__ == []
    assert cpp.__all__ == []
    assert gpu.__all__ == []
    assert metal.__all__ == []
