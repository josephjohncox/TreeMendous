"""
Unified Backend Manager for Tree-Mendous

Provides clean, protocol-based backend selection and management with proper dataclasses.
Eliminates protocol drift by enforcing consistent interfaces.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, Type, List, Union
from dataclasses import dataclass, field

# Import protocols
try:
    from treemendous.basic.protocols import (
        CoreIntervalManagerProtocol,
        EnhancedIntervalManagerProtocol,
        PerformanceTrackingProtocol,
        RandomizedProtocol,
        BackendConfiguration,
        ImplementationType,
        PerformanceTier,
        IntervalResult,
        AvailabilityStats,
        PerformanceStats,
        standardize_interval_result,
        standardize_intervals_list,
        standardize_availability_stats,
        standardize_performance_stats,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent / 'basic'))
    from protocols import (
        CoreIntervalManagerProtocol,
        EnhancedIntervalManagerProtocol,
        PerformanceTrackingProtocol,
        RandomizedProtocol,
        BackendConfiguration,
        ImplementationType,
        PerformanceTier,
        IntervalResult,
        AvailabilityStats,
        PerformanceStats,
        standardize_interval_result,
        standardize_intervals_list,
        standardize_availability_stats,
        standardize_performance_stats,
    )


@dataclass
class RuntimeBackendInfo:
    """Runtime information about a backend after loading and testing"""
    config: BackendConfiguration
    implementation_class: Type
    instance_created: bool = False
    basic_ops_working: bool = False
    enhanced_features_available: List[str] = field(default_factory=list)
    detected_features: List[str] = field(default_factory=list)
    performance_tier_confirmed: PerformanceTier = PerformanceTier.BASELINE


class UnifiedIntervalManager:
    """
    Unified wrapper that provides consistent interface over any backend implementation.
    
    This wrapper eliminates protocol drift by standardizing all return types and method signatures
    while delegating to the underlying implementation.
    """
    
    def __init__(self, backend_impl: Any, backend_info: RuntimeBackendInfo):
        self._impl = backend_impl
        self._info = backend_info
        self._operation_count = 0
    
    # Core protocol methods (standardized)
    def release_interval(self, start: int, end: int, data: Optional[Any] = None) -> None:
        """Add interval to available space"""
        self._operation_count += 1
        # Handle implementations that don't support data parameter
        try:
            return self._impl.release_interval(start, end, data)
        except TypeError:
            # Fallback for implementations without data support
            return self._impl.release_interval(start, end)
    
    def reserve_interval(self, start: int, end: int, data: Optional[Any] = None) -> None:
        """Remove interval from available space"""
        self._operation_count += 1
        # Handle implementations that don't support data parameter
        try:
            return self._impl.reserve_interval(start, end, data)
        except TypeError:
            # Fallback for implementations without data support
            return self._impl.reserve_interval(start, end)
    
    def find_interval(self, start: int, length: int) -> Optional[IntervalResult]:
        """Find available interval - returns standardized IntervalResult"""
        result = self._impl.find_interval(start, length)
        return standardize_interval_result(result)
    
    def get_intervals(self) -> List[IntervalResult]:
        """Get all available intervals - returns standardized list"""
        intervals = self._impl.get_intervals()
        return standardize_intervals_list(intervals)
    
    def get_total_available_length(self) -> int:
        """Get total available space"""
        return self._impl.get_total_available_length()
    
    # Enhanced methods (available if implementation supports them)
    def get_availability_stats(self) -> Optional[AvailabilityStats]:
        """Get comprehensive availability statistics"""
        if hasattr(self._impl, 'get_availability_stats'):
            stats = self._impl.get_availability_stats()
            return standardize_availability_stats(stats)
        return None
    
    def find_best_fit(self, length: int, prefer_early: bool = True) -> Optional[IntervalResult]:
        """Find best-fit interval"""
        if hasattr(self._impl, 'find_best_fit'):
            result = self._impl.find_best_fit(length, prefer_early)
            return standardize_interval_result(result)
        return None
    
    def find_largest_available(self) -> Optional[IntervalResult]:
        """Find largest available interval"""
        if hasattr(self._impl, 'find_largest_available'):
            result = self._impl.find_largest_available()
            return standardize_interval_result(result)
        return None
    
    def get_performance_stats(self) -> Optional[PerformanceStats]:
        """Get performance statistics"""
        if hasattr(self._impl, 'get_performance_stats'):
            stats = self._impl.get_performance_stats()
            perf_stats = standardize_performance_stats(stats)
            # Add our wrapper stats
            return PerformanceStats(
                operation_count=perf_stats.operation_count + self._operation_count,
                cache_hits=perf_stats.cache_hits,
                cache_hit_rate=perf_stats.cache_hit_rate,
                implementation_name=self._info.config.name,
                language=self._info.config.language
            )
        return PerformanceStats(
            operation_count=self._operation_count,
            implementation_name=self._info.config.name,
            language=self._info.config.language
        )
    
    # Randomized methods (available for treaps, etc.)
    def sample_random_interval(self) -> Optional[IntervalResult]:
        """Sample random interval (treaps only)"""
        if hasattr(self._impl, 'sample_random_interval'):
            result = self._impl.sample_random_interval()
            return standardize_interval_result(result)
        return None
    
    def verify_properties(self) -> bool:
        """Verify implementation-specific properties"""
        if hasattr(self._impl, 'verify_treap_properties'):
            return self._impl.verify_treap_properties()
        if hasattr(self._impl, 'verify_properties'):
            return self._impl.verify_properties()
        return True  # Assume valid if no verification method
    
    # Metadata access
    def get_backend_info(self) -> RuntimeBackendInfo:
        """Get backend information"""
        return self._info
    
    def get_implementation_type(self) -> ImplementationType:
        """Get implementation type"""
        return self._info.config.implementation_type
    
    def get_performance_tier(self) -> PerformanceTier:
        """Get performance tier"""
        return self._info.config.performance_tier
    
    def supports_feature(self, feature: str) -> bool:
        """Check if implementation supports a specific feature"""
        return feature in self._info.config.features
    
    # Direct access to underlying implementation (for advanced usage)
    def get_raw_implementation(self) -> Any:
        """Get direct access to underlying implementation"""
        return self._impl


class TreeMendousBackendManager:
    """
    Manages all Tree-Mendous backend implementations with proper protocol enforcement.
    
    Discovers, tests, and provides unified access to all available implementations
    while ensuring consistent interfaces through protocol standardization.
    """
    
    def __init__(self):
        self._backend_configs: Dict[str, RuntimeBackendInfo] = {}
        self._discover_and_validate_backends()
    
    def _discover_and_validate_backends(self):
        """Discover and validate all available backend implementations"""
        
        # Python implementations
        self._register_python_backends()
        
        # C++ implementations
        self._register_cpp_backends()
        
        # Validate all registered backends
        self._validate_backends()
    
    def _register_python_backends(self):
        """Register and test Python implementations"""
        
        python_backends = [
            BackendConfiguration(
                implementation_id="py_boundary",
                name="Python Boundary Manager",
                language="Python",
                implementation_type=ImplementationType.BOUNDARY,
                performance_tier=PerformanceTier.BASELINE,
                features=["core-operations"],
                constructor_args={}
            ),
            BackendConfiguration(
                implementation_id="py_avl_earliest",
                name="Python AVL Earliest",
                language="Python",
                implementation_type=ImplementationType.AVL_TREE,
                performance_tier=PerformanceTier.OPTIMIZED,
                features=["core-operations", "self-balancing", "earliest-fit"],
                constructor_args={}
            ),
            BackendConfiguration(
                implementation_id="py_summary",
                name="Python Summary Tree",
                language="Python", 
                implementation_type=ImplementationType.SUMMARY_TREE,
                performance_tier=PerformanceTier.OPTIMIZED,
                features=["core-operations", "summary-stats", "best-fit", "analytics"],
                constructor_args={}
            ),
            BackendConfiguration(
                implementation_id="py_treap",
                name="Python Treap",
                language="Python",
                implementation_type=ImplementationType.TREAP,
                performance_tier=PerformanceTier.OPTIMIZED,
                features=["core-operations", "probabilistic-balance", "random-sampling"],
                constructor_args={"random_seed": 42}
            ),
            BackendConfiguration(
                implementation_id="py_boundary_summary",
                name="Python Boundary Summary",
                language="Python",
                implementation_type=ImplementationType.BOUNDARY,
                performance_tier=PerformanceTier.OPTIMIZED,
                features=["core-operations", "summary-stats", "caching", "best-fit"],
                constructor_args={}
            ),
        ]
        
        for config in python_backends:
            self._try_register_backend(config, self._load_python_implementation)
    
    def _register_cpp_backends(self):
        """Register and test C++ implementations"""
        
        cpp_backends = [
            BackendConfiguration(
                implementation_id="cpp_boundary",
                name="C++ Boundary Manager",
                language="C++",
                implementation_type=ImplementationType.BOUNDARY,
                performance_tier=PerformanceTier.HIGH_PERFORMANCE,
                features=["core-operations", "native-performance"],
                constructor_args={}
            ),
            BackendConfiguration(
                implementation_id="cpp_treap",
                name="C++ Treap",
                language="C++",
                implementation_type=ImplementationType.TREAP,
                performance_tier=PerformanceTier.HIGH_PERFORMANCE,
                features=["core-operations", "native-performance", "probabilistic-balance"],
                constructor_args={"seed": 42}  # C++ uses 'seed' not 'random_seed'
            ),
            BackendConfiguration(
                implementation_id="cpp_boundary_summary",
                name="C++ Boundary Summary",
                language="C++",
                implementation_type=ImplementationType.BOUNDARY,
                performance_tier=PerformanceTier.HIGH_PERFORMANCE,
                features=["core-operations", "native-performance", "summary-stats", "caching"],
                constructor_args={}
            ),
            BackendConfiguration(
                implementation_id="cpp_boundary_optimized",
                name="C++ Boundary (Optimized)",
                language="C++",
                implementation_type=ImplementationType.BOUNDARY,
                performance_tier=PerformanceTier.HIGH_PERFORMANCE,
                features=["core-operations", "native-performance", "small-vector", "branch-hints"],
                constructor_args={}
            ),
            BackendConfiguration(
                implementation_id="cpp_boundary_summary_optimized",
                name="C++ Boundary Summary (Optimized)",
                language="C++",
                implementation_type=ImplementationType.BOUNDARY,
                performance_tier=PerformanceTier.HIGH_PERFORMANCE,
                features=["core-operations", "native-performance", "summary-stats", "caching", "small-vector", "branch-hints"],
                constructor_args={}
            ),
        ]
        
        for config in cpp_backends:
            self._try_register_backend(config, self._load_cpp_implementation)
    
    def _try_register_backend(self, config: BackendConfiguration, loader_func):
        """Try to register a backend implementation"""
        try:
            impl_class = loader_func(config)
            if impl_class:
                runtime_info = RuntimeBackendInfo(
                    config=config,
                    implementation_class=impl_class
                )
                config.available = True
                self._backend_configs[config.implementation_id] = runtime_info
                
        except ImportError:
            # Implementation not available (C++ not compiled, etc.)
            pass
        except Exception as e:
            print(f"⚠️  Failed to register {config.implementation_id}: {e}")
    
    def _load_python_implementation(self, config: BackendConfiguration) -> Optional[Type]:
        """Load Python implementation"""
        module_map = {
            "py_boundary": ("treemendous.basic.boundary", "IntervalManager"),
            "py_avl_earliest": ("treemendous.basic.avl_earliest", "EarliestIntervalTree"),
            "py_summary": ("treemendous.basic.summary", "SummaryIntervalTree"),
            "py_treap": ("treemendous.basic.treap", "IntervalTreap"),
            "py_boundary_summary": ("treemendous.basic.boundary_summary", "BoundarySummaryManager"),
        }
        
        if config.implementation_id not in module_map:
            return None
        
        module_path, class_name = module_map[config.implementation_id]
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    
    def _load_cpp_implementation(self, config: BackendConfiguration) -> Optional[Type]:
        """Load C++ implementation"""
        module_map = {
            "cpp_boundary": ("treemendous.cpp.boundary", "IntervalManager"),
            "cpp_treap": ("treemendous.cpp.treap", "IntervalTreap"),
            "cpp_boundary_summary": ("treemendous.cpp.boundary_summary", "BoundarySummaryManager"),
            "cpp_boundary_optimized": ("treemendous.cpp.boundary_optimized", "IntervalManager"),
            "cpp_boundary_summary_optimized": ("treemendous.cpp.boundary_summary_optimized", "BoundarySummaryManager"),
        }
        
        if config.implementation_id not in module_map:
            return None
        
        module_path, class_name = module_map[config.implementation_id]
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    
    def _validate_backends(self):
        """Validate all registered backends by testing basic functionality"""
        for backend_id, runtime_info in self._backend_configs.items():
            try:
                # Test instantiation
                instance = self._create_raw_instance(backend_id)
                runtime_info.instance_created = True
                
                # Test basic operations
                instance.release_interval(0, 100)
                total = instance.get_total_available_length()
                if total == 100:
                    runtime_info.basic_ops_working = True
                
                # Detect enhanced features
                enhanced_features = []
                if hasattr(instance, 'get_availability_stats'):
                    enhanced_features.append("availability-stats")
                if hasattr(instance, 'find_best_fit'):
                    enhanced_features.append("best-fit")
                if hasattr(instance, 'sample_random_interval'):
                    enhanced_features.append("random-sampling")
                if hasattr(instance, 'get_performance_stats'):
                    enhanced_features.append("performance-tracking")
                
                runtime_info.detected_features = enhanced_features
                runtime_info.enhanced_features_available = enhanced_features
                
            except Exception as e:
                print(f"⚠️  Backend validation failed for {backend_id}: {e}")
                runtime_info.config.available = False
    
    def _create_raw_instance(self, backend_id: str) -> Any:
        """Create raw implementation instance"""
        if backend_id not in self._backend_configs:
            raise ValueError(f"Backend '{backend_id}' not found")
        
        runtime_info = self._backend_configs[backend_id]
        impl_class = runtime_info.implementation_class
        constructor_args = runtime_info.config.constructor_args
        
        return impl_class(**constructor_args)
    
    def create_manager(self, backend_id: Optional[str] = None) -> UnifiedIntervalManager:
        """
        Create unified interval manager with standardized interface.
        
        Args:
            backend_id: Specific backend to use, or None for auto-selection
            
        Returns:
            UnifiedIntervalManager with consistent interface
        """
        if backend_id is None:
            backend_id = self.select_best_backend()
        
        if backend_id not in self._backend_configs:
            raise ValueError(f"Backend '{backend_id}' not available")
        
        runtime_info = self._backend_configs[backend_id]
        raw_impl = self._create_raw_instance(backend_id)
        
        return UnifiedIntervalManager(raw_impl, runtime_info)
    
    def select_best_backend(self) -> str:
        """Auto-select the best available backend"""
        available = self.get_available_backends()
        
        if not available:
            raise RuntimeError("No Tree-Mendous backends available")
        
        # Priority order: C++ Boundary Summary > C++ Treap > Python Boundary Summary > etc.
        priority_order = [
            "cpp_boundary_summary",
            "cpp_treap", 
            "cpp_boundary",
            "py_boundary_summary",
            "py_summary",
            "py_treap",
            "py_avl_earliest",
            "py_boundary",
        ]
        
        for backend_id in priority_order:
            if backend_id in available:
                return backend_id
        
        # Fallback to first available
        return list(available.keys())[0]
    
    def get_available_backends(self) -> Dict[str, RuntimeBackendInfo]:
        """Get all available and working backends"""
        return {
            k: v for k, v in self._backend_configs.items() 
            if v.config.available and v.basic_ops_working
        }
    
    def get_backends_by_type(self, impl_type: ImplementationType) -> Dict[str, RuntimeBackendInfo]:
        """Get backends by implementation type"""
        available = self.get_available_backends()
        return {
            k: v for k, v in available.items()
            if v.config.implementation_type == impl_type
        }
    
    def get_backends_by_language(self, language: str) -> Dict[str, RuntimeBackendInfo]:
        """Get backends by language"""
        available = self.get_available_backends()
        return {
            k: v for k, v in available.items()
            if v.config.language.lower() == language.lower()
        }
    
    def get_backends_with_feature(self, feature: str) -> Dict[str, RuntimeBackendInfo]:
        """Get backends that support a specific feature"""
        available = self.get_available_backends()
        return {
            k: v for k, v in available.items()
            if feature in v.detected_features or feature in v.config.features
        }
    
    def print_backend_status(self):
        """Print comprehensive backend status"""
        print("🌳 Tree-Mendous Backend Status")
        print("=" * 50)
        
        available = self.get_available_backends()
        unavailable = {k: v for k, v in self._backend_configs.items() if not v.config.available}
        
        print(f"✅ Available Backends ({len(available)}):")
        for backend_id, info in available.items():
            features_str = ", ".join(info.detected_features)
            print(f"  • {info.config.name}")
            print(f"    Language: {info.config.language}")
            print(f"    Type: {info.config.implementation_type.value}")
            print(f"    Tier: {info.config.performance_tier.value}")
            print(f"    Features: {features_str}")
            print()
        
        if unavailable:
            print(f"❌ Unavailable Backends ({len(unavailable)}):")
            for backend_id, info in unavailable.items():
                print(f"  • {info.config.name} ({info.config.language})")
        
        # Language breakdown
        python_count = len(self.get_backends_by_language("Python"))
        cpp_count = len(self.get_backends_by_language("C++"))
        print(f"📊 Language Breakdown: {python_count} Python, {cpp_count} C++")
        
        # Feature matrix
        all_features = set()
        for info in available.values():
            all_features.update(info.detected_features)
        
        if all_features:
            print(f"\n🎯 Feature Matrix:")
            print(f"{'Backend':<25} {'Features'}")
            print("-" * 50)
            for backend_id, info in available.items():
                features_icons = []
                for feature in sorted(all_features):
                    if feature in info.detected_features:
                        features_icons.append("✅")
                    else:
                        features_icons.append("❌")
                print(f"{info.config.name[:24]:<25} {' '.join(features_icons)}")


# Global singleton
_global_backend_manager: Optional[TreeMendousBackendManager] = None


def get_backend_manager() -> TreeMendousBackendManager:
    """Get global backend manager singleton"""
    global _global_backend_manager
    if _global_backend_manager is None:
        _global_backend_manager = TreeMendousBackendManager()
    return _global_backend_manager


def create_interval_tree(backend: Optional[str] = None) -> UnifiedIntervalManager:
    """
    Create interval tree with unified interface.
    
    Args:
        backend: Backend ID or None for auto-selection
        
    Returns:
        UnifiedIntervalManager with standardized protocol
    """
    manager = get_backend_manager()
    return manager.create_manager(backend)


def list_available_backends() -> Dict[str, RuntimeBackendInfo]:
    """List all available backends"""
    manager = get_backend_manager()
    return manager.get_available_backends()


def print_backend_status():
    """Print comprehensive backend status"""
    manager = get_backend_manager()
    manager.print_backend_status()
