#!/usr/bin/env python3
"""
Simple Boundary Summary Tests

Basic functionality tests for boundary summary implementations without external dependencies.
"""

import sys
import time
from pathlib import Path

# Add paths for import resolution  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'treemendous' / 'basic'))

try:
    from boundary_summary import BoundarySummaryManager
    print("✅ Python Boundary Summary loaded successfully")
    PY_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to load Python Boundary Summary: {e}")
    PY_AVAILABLE = False

try:
    from treemendous.cpp.boundary_summary import BoundarySummaryManager as CppBoundarySummaryManager
    print("✅ C++ Boundary Summary loaded successfully")
    CPP_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  C++ Boundary Summary not available: {e}")
    CPP_AVAILABLE = False


def test_basic_functionality():
    """Test basic boundary summary functionality"""
    print("Testing basic boundary summary functionality...")
    
    implementations = []
    if PY_AVAILABLE:
        implementations.append(("Python", BoundarySummaryManager()))
    if CPP_AVAILABLE:
        implementations.append(("C++", CppBoundarySummaryManager()))
    
    if not implementations:
        print("❌ No implementations available")
        return False
    
    for impl_name, manager in implementations:
        print(f"\n  Testing {impl_name} implementation:")
        
        # Test empty manager
        summary = manager.get_summary()
        assert summary.total_free_length == 0
        assert summary.interval_count == 0
        print(f"    ✓ Empty manager works correctly")
        
        # Test single interval
        manager.release_interval(10, 20)
        summary = manager.get_summary()
        assert summary.total_free_length == 10
        assert summary.interval_count == 1
        assert summary.largest_interval_length == 10
        print(f"    ✓ Single interval works correctly")
        
        # Test multiple intervals
        manager.release_interval(30, 50)
        manager.release_interval(60, 80)
        summary = manager.get_summary()
        assert summary.total_free_length == 10 + 20 + 20  # 50 total
        assert summary.interval_count == 3
        assert summary.largest_interval_length == 20
        print(f"    ✓ Multiple intervals work correctly")
        
        # Test advanced queries
        stats = manager.get_availability_stats()
        if hasattr(stats, 'total_free'):  # C++ object
            total_free = stats.total_free
            fragmentation = stats.fragmentation
        else:  # Python dict
            total_free = stats['total_free']
            fragmentation = stats['fragmentation']
        
        assert total_free == 50
        assert fragmentation >= 0.0 and fragmentation <= 1.0
        print(f"    ✓ Advanced queries work correctly")


def test_summary_analytics():
    """Test comprehensive summary analytics"""
    print("\nTesting summary analytics...")
    
    implementations = []
    if PY_AVAILABLE:
        implementations.append(("Python", BoundarySummaryManager()))
    if CPP_AVAILABLE:
        implementations.append(("C++", CppBoundarySummaryManager()))
    
    for impl_name, manager in implementations:
        print(f"\n  Testing {impl_name} analytics:")
        
        # Create fragmented scenario
        manager.release_interval(0, 1000)
        manager.reserve_interval(100, 200)  # Creates [0,100), [200,1000)
        manager.reserve_interval(300, 400)  # Creates [0,100), [200,300), [400,1000)
        manager.reserve_interval(500, 600)  # Creates [0,100), [200,300), [400,500), [600,1000)
        
        summary = manager.get_summary()
        stats = manager.get_availability_stats()
        
        # Verify fragmentation calculation
        expected_total = 100 + 100 + 100 + 400  # 700 total
        expected_largest = 400  # [600, 1000)
        expected_fragmentation = 1.0 - (400 / 700)  # ≈ 0.43
        
        assert summary.total_free_length == expected_total
        assert summary.largest_interval_length == expected_largest
        assert abs(summary.fragmentation_index - expected_fragmentation) < 0.01
        
        print(f"    ✓ Fragmentation: {summary.fragmentation_index:.2f} (expected ≈ {expected_fragmentation:.2f})")
        print(f"    ✓ Total free: {summary.total_free_length} (expected {expected_total})")
        print(f"    ✓ Largest: {summary.largest_interval_length} (expected {expected_largest})")


def test_best_fit_queries():
    """Test best-fit query functionality"""
    print("\nTesting best-fit queries...")
    
    implementations = []
    if PY_AVAILABLE:
        implementations.append(("Python", BoundarySummaryManager()))
    if CPP_AVAILABLE:
        implementations.append(("C++", CppBoundarySummaryManager()))
    
    for impl_name, manager in implementations:
        print(f"\n  Testing {impl_name} best-fit:")
        
        # Setup scenario with different sized gaps
        manager.release_interval(0, 1000)
        manager.reserve_interval(200, 300)  # Creates [0,200), [300,1000)
        
        # Test best-fit allocation
        best_fit = manager.find_best_fit(50)
        assert best_fit is not None, f"{impl_name}: Should find 50-unit interval"
        
        start, end = best_fit
        assert end - start == 50, f"{impl_name}: Best fit should be exactly 50 units"
        assert start >= 0, f"{impl_name}: Start should be non-negative"
        
        print(f"    ✓ Best fit (50 units): [{start}, {end})")
        
        # Test largest available
        largest = manager.find_largest_available()
        assert largest is not None, f"{impl_name}: Should find largest interval"
        
        start, end = largest
        expected_size = 700  # [300, 1000) should be largest
        assert end - start == expected_size, f"{impl_name}: Largest should be {expected_size}, got {end - start}"
        
        print(f"    ✓ Largest available: [{start}, {end}), size={end-start}")
    


def test_performance_characteristics():
    """Test performance characteristics"""
    print("\nTesting performance characteristics...")
    
    implementations = []
    if PY_AVAILABLE:
        implementations.append(("Python", BoundarySummaryManager()))
    if CPP_AVAILABLE:
        implementations.append(("C++", CppBoundarySummaryManager()))
    
    for impl_name, manager in implementations:
        print(f"\n  Testing {impl_name} performance:")
        
        # Setup
        manager.release_interval(0, 10000)
        
        # Time basic operations
        num_ops = 1000
        operations = []
        for i in range(num_ops):
            op = 'reserve' if i % 2 == 0 else 'release'
            start = (i * 17) % 9000
            end = start + ((i * 13) % 500) + 1
            operations.append((op, start, end))
        
        start_time = time.perf_counter()
        for op, start, end in operations:
            if op == 'reserve':
                manager.reserve_interval(start, end)
            else:
                manager.release_interval(start, end)
        operation_time = time.perf_counter() - start_time
        
        # Time summary access (should be O(1) with caching)
        start_time = time.perf_counter()
        for _ in range(100):
            if hasattr(manager, 'get_availability_stats'):
                manager.get_availability_stats()
            else:
                manager.get_summary()
        summary_time = time.perf_counter() - start_time
        
        ops_per_second = num_ops / operation_time if operation_time > 0 else 0
        avg_summary_time = summary_time / 100 * 1_000_000  # microseconds
        
        print(f"    Operations: {ops_per_second:,.0f} ops/sec")
        print(f"    Summary access: {avg_summary_time:.1f}µs average")
        
        # Check caching performance
        if hasattr(manager, 'get_performance_stats'):
            perf = manager.get_performance_stats()
            if hasattr(perf, 'cache_hit_rate'):  # C++ object
                print(f"    Cache hit rate: {perf.cache_hit_rate:.1%}")
            else:  # Python dict
                print(f"    Cache hit rate: {perf['cache_hit_rate']:.1%}")
        
        # Performance should be reasonable
        assert ops_per_second > 1000, f"{impl_name}: Performance too slow: {ops_per_second:.0f} ops/sec"
        assert avg_summary_time < 100, f"{impl_name}: Summary access too slow: {avg_summary_time:.1f}µs"
    


def test_py_cpp_equivalence():
    """Test Python vs C++ equivalence"""
    if not (PY_AVAILABLE and CPP_AVAILABLE):
        print("⚠️  Cannot test equivalence - missing implementation")
    
    print("\nTesting Python vs C++ equivalence...")
    
    py_manager = BoundarySummaryManager()
    cpp_manager = CppBoundarySummaryManager()
    
    # Apply same operations
    operations = [
        ('release', 0, 1000),
        ('reserve', 100, 200),
        ('reserve', 300, 400),
        ('release', 150, 350),
        ('reserve', 600, 700),
    ]
    
    for op, start, end in operations:
        if op == 'reserve':
            py_manager.reserve_interval(start, end)
            cpp_manager.reserve_interval(start, end)
        else:
            py_manager.release_interval(start, end)
            cpp_manager.release_interval(start, end)
    
    # Compare results
    py_stats = py_manager.get_availability_stats()
    cpp_stats = cpp_manager.get_availability_stats()
    
    # Core metrics should match
    assert py_stats['total_free'] == cpp_stats.total_free
    assert py_stats['free_chunks'] == cpp_stats.free_chunks
    assert py_stats['largest_chunk'] == cpp_stats.largest_chunk
    
    # Fragmentation should be close
    frag_diff = abs(py_stats['fragmentation'] - cpp_stats.fragmentation)
    assert frag_diff < 0.001, f"Fragmentation differs: {frag_diff}"
    
    print("    ✓ Python and C++ implementations produce identical results")
    


def test_caching_behavior():
    """Test summary caching behavior"""
    if not PY_AVAILABLE:
        print("⚠️  Python implementation not available for caching test")
    
    print("\nTesting caching behavior...")
    
    manager = BoundarySummaryManager()
    manager.release_interval(0, 1000)
    
    # First access should compute summary
    summary1 = manager.get_summary()
    perf1 = manager.get_performance_stats()
    
    # Repeated access should use cache
    for _ in range(5):
        summary2 = manager.get_summary()
    
    perf2 = manager.get_performance_stats()
    
    # Cache hits should increase
    assert perf2['cache_hits'] > perf1['cache_hits'], "Cache hits should increase"
    
    # Modify manager to invalidate cache
    manager.reserve_interval(100, 200)
    
    # Next access should recompute
    summary3 = manager.get_summary()
    assert summary3.total_free_length != summary1.total_free_length, "Summary should update after modification"
    
    print("    ✓ Caching works correctly")
    


def main():
    """Run all boundary summary tests"""
    print("🏗️ Tree-Mendous Boundary Summary Tests")
    print("Testing boundary-based summary interval managers")
    print("=" * 55)
    
    if not (PY_AVAILABLE or CPP_AVAILABLE):
        print("❌ No boundary summary implementations available")
        return False
    
    tests = [
        test_basic_functionality,
        test_summary_analytics,
        test_best_fit_queries,
        test_performance_characteristics,
        test_py_cpp_equivalence,
        test_caching_behavior,
    ]
    
    passed = 0
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 55}")
    print(f" Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print(" 🎉 All boundary summary tests passed!")
        print(f"\n📊 Key features verified:")
        print(f"   • Boundary-based efficiency with summary analytics")
        print(f"   • O(1) cached summary statistics")
        print(f"   • Best-fit and largest-available queries")
        print(f"   • Python/C++ implementation equivalence")
        print(f"   • Performance optimization with caching")
        
        # Show available implementations
        impls = []
        if PY_AVAILABLE:
            impls.append("Python")
        if CPP_AVAILABLE:
            impls.append("C++")
        
        print(f"   • Available implementations: {', '.join(impls)}")
        
    else:
        print(" ❌ Some tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
