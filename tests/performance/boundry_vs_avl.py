from typing import List, Tuple, Callable, Dict, Type, Protocol, Any
from dataclasses import dataclass
from collections import defaultdict
import time
import random
from treemendous.basic.base import IntervalManagerProtocol
from treemendous.cpp.boundary import IntervalManager as CppIntervalManager
from treemendous.basic.boundary import IntervalManager as PyIntervalManager
from treemendous.basic.avl_earliest import EarliestIntervalTree
from tests.performance.workload import generate_realistic_workload

# Try to import CppICIntervalManager, set to None if import fails
try:
    from treemendous.cpp.boundary import ICIntervalManager as CppICIntervalManager
except ImportError:
    CppICIntervalManager = None

INITIAL_INTERVAL_SIZE: Tuple[int, int] = (0, 10_000_000)
ITERATIONS: int = 100_000

@dataclass
class BenchmarkResult:
    total_time: float
    op_times: Dict[str, float]
    intervals: List[Tuple[int, int]]


def normalize_intervals(intervals: List[Any]) -> List[Tuple[int, int]]:
    """Normalize interval list to (start, end) tuples."""
    normalized: List[Tuple[int, int]] = []
    for interval in intervals:
        if hasattr(interval, 'start') and hasattr(interval, 'end'):
            normalized.append((interval.start, interval.end))
        elif isinstance(interval, tuple) and len(interval) >= 2:
            normalized.append((interval[0], interval[1]))
    return normalized


def benchmark_manager(
    manager_class: Type[IntervalManagerProtocol], 
    operations: List[Tuple[str, int, int]]
) -> BenchmarkResult:
    manager = manager_class()
    manager.release_interval(*INITIAL_INTERVAL_SIZE)
    op_times: Dict[str, List[float]] = defaultdict(list)
    
    for op, start, end in operations:
        start_time = time.time()
        if op == 'reserve':
            manager.reserve_interval(start, end)
        elif op == 'release':
            manager.release_interval(start, end)
        elif op == 'find':
            manager.find_interval(start, end - start)
        op_times[op].append(time.time() - start_time)
    
    avg_times = {op: sum(times) / len(times) for op, times in op_times.items()}
    return BenchmarkResult(
        total_time=sum(sum(times) for times in op_times.values()),
        op_times=avg_times,
        intervals=sorted(normalize_intervals(manager.get_intervals()))
    )

def generate_operations(num_operations: int) -> List[Tuple[str, int, int]]:
    return generate_realistic_workload(
        num_operations=num_operations,
        profile="allocator",
        space_range=(0, 9_999_900),
        operation_mix={'reserve': 0.4, 'release': 0.4, 'find': 0.2},
        seed=42,
        include_data=False
    )

def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    merged: List[Tuple[int, int]] = []
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start == current_end:  # Only merge if exactly adjacent
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged

def run_benchmarks(random_seed: int | None) -> None:
    if random_seed:
        random.seed(random_seed)
    operations = generate_operations(ITERATIONS)

    managers = {
        "C++ Boundary": CppIntervalManager,
        "Python Boundary": PyIntervalManager,
        "AVL Tree": EarliestIntervalTree
    }
    
    # Only add CppICIntervalManager if import succeeded
    if CppICIntervalManager is not None:
        managers["C++ IC Boundary"] = CppICIntervalManager

    results = {
        name: benchmark_manager(manager_class, operations)
        for name, manager_class in managers.items()
    }

    # Print results and comparisons
    for name, result in results.items():
        print(f"\n{name} execution time: {result.total_time:.4f} seconds")
        print("Average times per operation:")
        for op in ['reserve', 'release', 'find']:
            if op in result.op_times:
                print(f"  {op:8}: {result.op_times[op]*1000:.4f} ms")
    
    # Calculate and display speedups relative to C++ Boundary
    cpp_boundary_time = results["C++ Boundary"].total_time
    print("\nSpeedup relative to C++ Boundary:")
    for name, result in results.items():
        if name != "C++ Boundary":
            speedup = result.total_time / cpp_boundary_time
            print(f"  {name:15}: {speedup:.2f}x slower")

    # Compare results for correctness
    base_intervals = merge_intervals(results["C++ Boundary"].intervals)
    for name, result in results.items():
        if name != "C++ Boundary":
            result_intervals = merge_intervals(result.intervals)
            are_equivalent = result_intervals == base_intervals
            print(f"\n{name} matches C++ Boundary: {are_equivalent}")
            
            if not are_equivalent:
                print(f"Effective coverage differs for {name}:")
                print("Base coverage:     ", end="")
                print("Result coverage:   ", end="")
                print()
                
                base_diffs = [x for x in base_intervals if x not in result_intervals]
                result_diffs = [x for x in result_intervals if x not in base_intervals]
                
                for i in range(max(len(base_diffs), len(result_diffs))):
                    base_interval = f"[{base_diffs[i][0]}, {base_diffs[i][1]}]*" if i < len(base_diffs) else " " * 20
                    result_interval = f"[{result_diffs[i][0]}, {result_diffs[i][1]}]*" if i < len(result_diffs) else ""
                    print(f"{base_interval:20} {result_interval}")

if __name__ == "__main__":
    import sys
    run_benchmarks(random_seed=int(sys.argv[1]) if len(sys.argv) > 1 else None)
