"""Compatibility facade over the typed immutable backend catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from treemendous.backends import (
    CATALOG,
    CATALOG_BY_ID,
    Available,
    BackendRequest,
    BackendSpec,
    Capability,
    Invalid,
    Runtime,
    Unavailable,
    probe_backend,
    select_backend,
)
from treemendous.backends.adapters import CppBackendAdapter, PythonBackendAdapter
from treemendous.basic.protocols import (
    AvailabilityStats,
    BackendConfiguration,
    ImplementationType,
    IntervalResult,
    PerformanceStats,
    PerformanceTier,
    standardize_availability_stats,
    standardize_interval_result,
    standardize_performance_stats,
)
from treemendous.domain import BackendInvalidError, BackendUnavailableError, Span
from treemendous.policies import PayloadPolicy
from treemendous.rangeset import RangeSet

_CAPABILITY_NAMES = {
    Capability.CORE: "core-operations",
    Capability.PAYLOADS: "payloads",
    Capability.ANALYTICS: "summary-stats",
    Capability.BEST_FIT: "best-fit",
    Capability.RANDOM_SAMPLE: "random-sampling",
    Capability.ATOMIC_ALLOCATE: "atomic-allocate",
}

# Deprecated string metadata remains a compatibility view only. Selection and
# semantic probing continue to use BackendSpec.capabilities, never these labels.
_LEGACY_FEATURES: dict[str, tuple[str, ...]] = {
    "py_boundary": ("core-operations",),
    "py_avl_earliest": ("core-operations", "self-balancing", "earliest-fit"),
    "py_summary": ("core-operations", "summary-stats", "best-fit", "analytics"),
    "py_treap": (
        "core-operations",
        "probabilistic-balance",
        "random-sampling",
    ),
    "py_boundary_summary": (
        "core-operations",
        "summary-stats",
        "caching",
        "best-fit",
        "analytics",
    ),
    "cpp_boundary": ("core-operations", "native-performance"),
    "cpp_treap": (
        "core-operations",
        "native-performance",
        "probabilistic-balance",
    ),
    "cpp_boundary_summary": (
        "core-operations",
        "native-performance",
        "summary-stats",
        "caching",
    ),
    "cpp_boundary_optimized": (
        "core-operations",
        "native-performance",
        "small-vector",
        "branch-hints",
    ),
    "cpp_boundary_summary_optimized": (
        "core-operations",
        "native-performance",
        "summary-stats",
        "caching",
        "small-vector",
        "branch-hints",
    ),
    "gpu_boundary_summary": (
        "core-operations",
        "gpu-accelerated",
        "parallel-reduction",
        "summary-stats",
        "massive-parallelism",
        "O(1) analytics",
    ),
    "metal_boundary_summary": (
        "core-operations",
        "gpu-accelerated",
        "metal-performance-shaders",
        "summary-stats",
        "apple-silicon",
        "O(1) analytics",
    ),
    "metal_boundary_summary_mixed": (
        "core-operations",
        "mixed-policy",
        "gpu-best-fit",
        "cpu-summary",
        "apple-silicon",
        "cached-analytics",
    ),
}

_DETECTED_METHOD_FEATURES = {
    "get_availability_stats": "availability-stats",
    "find_best_fit": "best-fit",
    "sample_random_interval": "random-sampling",
    "get_performance_stats": "performance-tracking",
}


def _implementation_type(spec: BackendSpec) -> ImplementationType:
    return {
        "boundary": ImplementationType.BOUNDARY,
        "avl": ImplementationType.AVL_TREE,
        "summary": ImplementationType.SUMMARY_TREE,
        "treap": ImplementationType.TREAP,
    }[spec.algorithm.value]


@dataclass(frozen=True)
class RuntimeBackendInfo:
    config: BackendConfiguration
    implementation_class: type[Any]
    instance_created: bool
    basic_ops_working: bool
    enhanced_features_available: tuple[str, ...]
    detected_features: tuple[str, ...]
    performance_tier_confirmed: PerformanceTier
    probe_reason: str | None = None


class UnifiedIntervalManager(RangeSet):
    """Legacy method surface backed by one explicit adapter."""

    def __init__(
        self, adapter, info: RuntimeBackendInfo, capabilities: frozenset[Capability]
    ):
        super().__init__(adapter, capabilities=capabilities, initially_available=False)
        self._info = info
        self._operation_count = 0

    def release_interval(self, start: int, end: int, data: Any = None) -> None:
        span = Span(start, end)
        with self._lock:
            self.add(span) if data is None else self.add(span, data)
            self._operation_count += 1

    def reserve_interval(self, start: int, end: int, data: Any = None) -> None:
        del data
        span = Span(start, end)
        with self._lock:
            self.discard(span)
            self._operation_count += 1

    def get_availability_stats(self) -> AvailabilityStats | None:
        """Return legacy raw-backend analytics without requiring a domain."""
        with self._lock:
            raw = self._adapter.implementation
            if not hasattr(raw, "get_availability_stats"):
                return None
            return standardize_availability_stats(raw.get_availability_stats())

    def find_best_fit(
        self, length: int, prefer_early: bool = True
    ) -> IntervalResult | None:
        with self._lock:
            raw = self._adapter.implementation
            if not hasattr(raw, "find_best_fit"):
                return None
            return standardize_interval_result(raw.find_best_fit(length, prefer_early))

    def find_largest_available(self) -> IntervalResult | None:
        with self._lock:
            raw = self._adapter.implementation
            if not hasattr(raw, "find_largest_available"):
                return None
            return standardize_interval_result(raw.find_largest_available())

    def sample_random_interval(self) -> IntervalResult | None:
        with self._lock:
            raw = self._adapter.implementation
            if not hasattr(raw, "sample_random_interval"):
                return None
            return standardize_interval_result(raw.sample_random_interval())

    def verify_properties(self) -> bool:
        with self._lock:
            raw = self._adapter.implementation
            verifier = getattr(raw, "verify_properties", None) or getattr(
                raw, "verify_treap_properties", None
            )
            return verifier() if verifier else True

    def get_performance_stats(self) -> PerformanceStats:
        with self._lock:
            raw = self._adapter.implementation
            if hasattr(raw, "get_performance_stats"):
                stats = standardize_performance_stats(raw.get_performance_stats())
                return PerformanceStats(
                    operation_count=stats.operation_count + self._operation_count,
                    cache_hits=stats.cache_hits,
                    implementation_name=self._info.config.name,
                    language=self._info.config.language,
                )
            return PerformanceStats(
                operation_count=self._operation_count,
                implementation_name=self._info.config.name,
                language=self._info.config.language,
            )

    def get_backend_info(self) -> RuntimeBackendInfo:
        return self._info

    def get_implementation_type(self) -> ImplementationType:
        return self._info.config.implementation_type

    def get_performance_tier(self) -> PerformanceTier:
        return self._info.config.performance_tier

    def supports_feature(self, feature: str) -> bool:
        return (
            feature in self._info.config.features
            or feature in self._info.detected_features
        )


class TreeMendousBackendManager:
    """Loads, semantically probes, and selects immutable backend specs."""

    def __init__(self) -> None:
        self._probes = {spec.id: probe_backend(spec) for spec in CATALOG}
        self._infos = {spec.id: self._info(spec) for spec in CATALOG}

    def _info(self, spec: BackendSpec) -> RuntimeBackendInfo:
        probe = self._probes[spec.id]
        available = isinstance(probe, Available)
        capability_features = tuple(
            sorted(_CAPABILITY_NAMES[cap] for cap in spec.capabilities)
        )
        features = tuple(
            dict.fromkeys((*capability_features, *_LEGACY_FEATURES.get(spec.id, ())))
        )
        try:
            implementation_class = spec.loader()
        except Exception:
            implementation_class = object
        detected_features = tuple(
            label
            for method, label in _DETECTED_METHOD_FEATURES.items()
            if hasattr(implementation_class, method)
        )
        reason = (
            None
            if available
            else getattr(probe, "reason", getattr(probe, "error", None))
        )
        if spec.id == "cpp_boundary_optimized":
            tier = PerformanceTier.BASELINE
        elif spec.runtime is not Runtime.PYTHON:
            tier = PerformanceTier.HIGH_PERFORMANCE
        else:
            tier = PerformanceTier.OPTIMIZED
        config = BackendConfiguration(
            implementation_id=spec.id,
            name=spec.name,
            language=spec.runtime.value,
            implementation_type=_implementation_type(spec),
            performance_tier=tier,
            features=features,
            available=available,
            constructor_args=spec.constructor_args,
        )
        return RuntimeBackendInfo(
            config,
            implementation_class,
            available,
            available,
            detected_features,
            detected_features,
            tier,
            reason,
        )

    def _spec(self, backend_id: str) -> BackendSpec:
        try:
            return CATALOG_BY_ID[backend_id]
        except KeyError as exc:
            raise BackendUnavailableError(f"unknown backend: {backend_id}") from exc

    def _require_available(self, spec: BackendSpec) -> None:
        probe = self._probes[spec.id]
        if isinstance(probe, Invalid):
            raise BackendInvalidError(
                f"backend {spec.id} failed validation: {probe.error}"
            )
        if isinstance(probe, Unavailable):
            raise BackendUnavailableError(
                f"backend {spec.id} unavailable: {probe.reason}"
            )

    @staticmethod
    def _adapter(spec: BackendSpec, implementation: Any):
        if spec.runtime is Runtime.PYTHON:
            return PythonBackendAdapter(implementation)
        return CppBackendAdapter(implementation)

    def create_manager(
        self, backend_id: str | None = None, **constructor_options: Any
    ) -> UnifiedIntervalManager:
        if backend_id is None or backend_id == "auto":
            spec = select_backend(CATALOG, self._probes, BackendRequest()).selected
        else:
            spec = self._spec(backend_id)
            self._require_available(spec)
        args = dict(spec.constructor_args)
        args.update(constructor_options)
        implementation = spec.loader()(**args)
        return UnifiedIntervalManager(
            self._adapter(spec, implementation), self._infos[spec.id], spec.capabilities
        )

    def create_range_set(
        self,
        domain,
        backend_id: str | None = None,
        require: frozenset[Capability] = frozenset({Capability.CORE}),
        payload_policy: PayloadPolicy[Any] | None = None,
        initially_available: bool = True,
        **constructor_options: Any,
    ) -> RangeSet:
        if backend_id is None or backend_id == "auto":
            spec = select_backend(
                CATALOG, self._probes, BackendRequest(require=require)
            ).selected
        else:
            spec = self._spec(backend_id)
            self._require_available(spec)
            if not require <= spec.capabilities:
                raise BackendUnavailableError(
                    f"backend {backend_id} lacks required capabilities"
                )
        args = dict(spec.constructor_args)
        args.update(constructor_options)
        implementation = spec.loader()(**args)
        return RangeSet(
            self._adapter(spec, implementation),
            domain=domain,
            capabilities=spec.capabilities,
            payload_policy=payload_policy,
            initially_available=initially_available,
        )

    def select_best_backend(self) -> str:
        return select_backend(CATALOG, self._probes, BackendRequest()).selected.id

    def get_available_backends(self) -> dict[str, RuntimeBackendInfo]:
        return {
            backend_id: info
            for backend_id, info in self._infos.items()
            if info.config.available
        }

    def get_backends_by_type(
        self, impl_type: ImplementationType
    ) -> dict[str, RuntimeBackendInfo]:
        return {
            key: info
            for key, info in self.get_available_backends().items()
            if info.config.implementation_type is impl_type
        }

    def get_backends_by_language(self, language: str) -> dict[str, RuntimeBackendInfo]:
        return {
            key: info
            for key, info in self.get_available_backends().items()
            if info.config.language.lower() == language.lower()
        }

    def get_backends_with_feature(self, feature: str) -> dict[str, RuntimeBackendInfo]:
        return {
            key: info
            for key, info in self.get_available_backends().items()
            if feature in info.config.features or feature in info.detected_features
        }

    def print_backend_status(self) -> None:
        for spec in CATALOG:
            info = self._infos[spec.id]
            status = (
                "available"
                if info.config.available
                else f"unavailable ({info.probe_reason})"
            )
            print(f"{spec.id}: {status}")


_global_backend_manager: TreeMendousBackendManager | None = None


def get_backend_manager() -> TreeMendousBackendManager:
    global _global_backend_manager
    if _global_backend_manager is None:
        _global_backend_manager = TreeMendousBackendManager()
    return _global_backend_manager


def create_interval_tree(
    backend: str | None = None, **options: Any
) -> UnifiedIntervalManager:
    return get_backend_manager().create_manager(backend, **options)


def create_range_set(
    domain,
    backend: str | None = None,
    require=frozenset({Capability.CORE}),
    payload_policy: PayloadPolicy[Any] | None = None,
    initially_available: bool = True,
    **options: Any,
) -> RangeSet:
    return get_backend_manager().create_range_set(
        domain,
        backend,
        require,
        payload_policy,
        initially_available,
        **options,
    )


def list_available_backends() -> dict[str, RuntimeBackendInfo]:
    return get_backend_manager().get_available_backends()


def print_backend_status() -> None:
    get_backend_manager().print_backend_status()
