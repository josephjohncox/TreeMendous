"""Single immutable production backend catalog."""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import Any

from .types import Algorithm, BackendSpec, Capability, Device, Maturity, Runtime


def _load(path: str, name: str) -> type[Any]:
    module = __import__(path, fromlist=[name])
    return getattr(module, name)


def _loader(path: str, name: str) -> Callable[[], type[Any]]:
    return lambda: _load(path, name)


CORE = frozenset({Capability.CORE, Capability.ATOMIC_ALLOCATE})
PAYLOAD_CORE = CORE | {Capability.PAYLOADS}

CATALOG: tuple[BackendSpec, ...] = (
    BackendSpec(
        "py_boundary",
        "Python Boundary",
        Algorithm.BOUNDARY,
        Runtime.PYTHON,
        Device.CPU,
        Maturity.STABLE,
        PAYLOAD_CORE,
        64,
        True,
        _loader("treemendous.basic.boundary", "IntervalManager"),
        MappingProxyType({}),
    ),
    BackendSpec(
        "py_avl_earliest",
        "Python AVL Earliest",
        Algorithm.AVL,
        Runtime.PYTHON,
        Device.CPU,
        Maturity.STABLE,
        PAYLOAD_CORE,
        64,
        True,
        _loader("treemendous.basic.avl_earliest", "EarliestIntervalTree"),
        MappingProxyType({}),
    ),
    BackendSpec(
        "py_summary",
        "Python Summary",
        Algorithm.SUMMARY,
        Runtime.PYTHON,
        Device.CPU,
        Maturity.STABLE,
        PAYLOAD_CORE | {Capability.ANALYTICS, Capability.BEST_FIT},
        64,
        True,
        _loader("treemendous.basic.summary", "SummaryIntervalTree"),
        MappingProxyType({}),
    ),
    BackendSpec(
        "py_treap",
        "Python Treap",
        Algorithm.TREAP,
        Runtime.PYTHON,
        Device.CPU,
        Maturity.STABLE,
        PAYLOAD_CORE | {Capability.RANDOM_SAMPLE},
        64,
        True,
        _loader("treemendous.basic.treap", "IntervalTreap"),
        MappingProxyType({"random_seed": 42}),
    ),
    BackendSpec(
        "py_boundary_summary",
        "Python Boundary Summary",
        Algorithm.BOUNDARY,
        Runtime.PYTHON,
        Device.CPU,
        Maturity.STABLE,
        PAYLOAD_CORE | {Capability.ANALYTICS, Capability.BEST_FIT},
        64,
        True,
        _loader("treemendous.basic.boundary_summary", "BoundarySummaryManager"),
        MappingProxyType({}),
    ),
    BackendSpec(
        "cpp_boundary",
        "C++ Boundary",
        Algorithm.BOUNDARY,
        Runtime.CPP,
        Device.CPU,
        Maturity.STABLE,
        CORE,
        64,
        True,
        _loader("treemendous.cpp.boundary", "IntervalManager"),
        MappingProxyType({}),
    ),
    BackendSpec(
        "cpp_treap",
        "C++ Treap (experimental 32-bit)",
        Algorithm.TREAP,
        Runtime.CPP,
        Device.CPU,
        Maturity.EXPERIMENTAL,
        frozenset(),
        32,
        True,
        _loader("treemendous.cpp.treap", "IntervalTreap"),
        MappingProxyType({"seed": 42}),
    ),
    BackendSpec(
        "cpp_boundary_summary",
        "C++ Boundary Summary (experimental 32-bit)",
        Algorithm.BOUNDARY,
        Runtime.CPP,
        Device.CPU,
        Maturity.EXPERIMENTAL,
        frozenset(),
        32,
        True,
        _loader("treemendous.cpp.boundary_summary", "BoundarySummaryManager"),
        MappingProxyType({}),
    ),
    BackendSpec(
        "cpp_boundary_optimized",
        "C++ Boundary Parity Alias",
        Algorithm.BOUNDARY,
        Runtime.CPP,
        Device.CPU,
        Maturity.STABLE,
        CORE,
        64,
        True,
        _loader("treemendous.cpp.boundary_optimized", "IntervalManager"),
        MappingProxyType({}),
    ),
    BackendSpec(
        "cpp_boundary_summary_optimized",
        "C++ Boundary Summary Optimized (experimental 32-bit)",
        Algorithm.BOUNDARY,
        Runtime.CPP,
        Device.CPU,
        Maturity.EXPERIMENTAL,
        frozenset(),
        32,
        True,
        _loader("treemendous.cpp.boundary_summary_optimized", "BoundarySummaryManager"),
        MappingProxyType({}),
    ),
    BackendSpec(
        "gpu_boundary_summary",
        "CUDA Boundary Summary",
        Algorithm.BOUNDARY,
        Runtime.CUDA,
        Device.GPU,
        Maturity.EXPERIMENTAL,
        frozenset(),
        32,
        False,
        _loader(
            "treemendous.cpp.gpu.boundary_summary_gpu", "GPUBoundarySummaryManager"
        ),
        MappingProxyType({}),
    ),
    BackendSpec(
        "metal_boundary_summary",
        "Metal Boundary Summary",
        Algorithm.BOUNDARY,
        Runtime.METAL,
        Device.GPU,
        Maturity.EXPERIMENTAL,
        frozenset(),
        32,
        False,
        _loader(
            "treemendous.cpp.metal.boundary_summary_metal",
            "MetalBoundarySummaryManager",
        ),
        MappingProxyType({}),
    ),
    BackendSpec(
        "metal_boundary_summary_mixed",
        "Metal Boundary Summary Mixed",
        Algorithm.BOUNDARY,
        Runtime.METAL,
        Device.GPU,
        Maturity.EXPERIMENTAL,
        frozenset(),
        32,
        False,
        _loader("treemendous.cpp.metal.mixed", "MixedBoundarySummaryManager"),
        MappingProxyType({}),
    ),
)

CATALOG_BY_ID = MappingProxyType({spec.id: spec for spec in CATALOG})
