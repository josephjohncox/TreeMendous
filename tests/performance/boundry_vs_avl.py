from typing import List, Tuple, Callable, Dict
from collections import defaultdict
import time
import random
from treemendous.cpp.boundary import IntervalManager
# from treemendous.basic.boundary import IntervalManager
from treemendous.basic.avl_earliest import EarliestIntervalTree

INITIAL_INTERVAL_SIZE: Tuple[int, int] = (0, 10_000_000)
ITERATIONS: int = 100_000

def benchmark_interval_manager(operations: List[Tuple[str, int, int]]) -> Tuple[float, Dict[str, float]]:
    manager: IntervalManager = IntervalManager()
    manager.release_interval(*INITIAL_INTERVAL_SIZE)
    op_times: Dict[str, List[float]] = defaultdict(list)
    
    for op, start, end in operations:
        start_time: float = time.time()
        if op == 'reserve':
            manager.reserve_interval(start, end)
        elif op == 'release':
            manager.release_interval(start, end)
        elif op == 'find':
            manager.find_interval(start, end - start)
        op_times[op].append(time.time() - start_time)
    
    avg_times: Dict[str, float] = {
        op: sum(times) / len(times) for op, times in op_times.items()
    }
    return sum(sum(times) for times in op_times.values()), avg_times

def benchmark_interval_tree(operations: List[Tuple[str, int, int]]) -> Tuple[float, Dict[str, float]]:
    tree: EarliestIntervalTree = EarliestIntervalTree()
    tree.insert_interval(*INITIAL_INTERVAL_SIZE)
    op_times: Dict[str, List[float]] = defaultdict(list)
    
    for op, start, end in operations:
        start_time: float = time.time()
        if op == 'reserve':
            tree.delete_interval(start, end)
        elif op == 'release':
            tree.insert_interval(start, end)
        elif op == 'find':
            tree.find_interval(start, end - start)
        op_times[op].append(time.time() - start_time)
    
    avg_times: Dict[str, float] = {
        op: sum(times) / len(times) for op, times in op_times.items()
    }
    return sum(sum(times) for times in op_times.values()), avg_times

def generate_operations(num_operations: int) -> List[Tuple[str, int, int]]:
    operations: List[Tuple[str, int, int]] = []
    for _ in range(num_operations):
        op_type: str = random.choice(['reserve', 'release', 'find'])
        start: int = random.randint(0, 9_999_899)
        length: int = random.randint(1, 1000)
        end: int = start + length
        operations.append((op_type, start, end))
    return operations

def get_intervals(manager: IntervalManager) -> List[Tuple[int, int]]:
    return sorted(manager.get_intervals())

def get_tree_intervals(tree: EarliestIntervalTree) -> List[Tuple[int, int]]:
    return sorted(tree.get_all_intervals())

def run_benchmarks(random_seed: int | None) -> None:
    if random_seed:
        random.seed(random_seed)
    operations: List[Tuple[str, int, int]] = generate_operations(ITERATIONS)

    # Create fresh instances for validation
    manager: IntervalManager = IntervalManager()
    tree: EarliestIntervalTree = EarliestIntervalTree()
    manager.release_interval(*INITIAL_INTERVAL_SIZE)
    tree.insert_interval(*INITIAL_INTERVAL_SIZE)

    time_manager, manager_op_times = benchmark_interval_manager(operations)
    time_tree, tree_op_times = benchmark_interval_tree(operations)

    print(f"\nIntervalManager execution time: {time_manager:.4f} seconds")
    print("Average times per operation:")
    for op in ['reserve', 'release', 'find']:
        if op in manager_op_times:
            print(f"  {op:8}: {manager_op_times[op]*1000:.4f} ms")

    print(f"\nEarliestIntervalTree execution time: {time_tree:.4f} seconds")
    print("Average times per operation:")
    for op in ['reserve', 'release', 'find']:
        if op in tree_op_times:
            print(f"  {op:8}: {tree_op_times[op]*1000:.4f} ms")

    # Run operations again to check final state
    for op, start, end in operations:
        if op == 'reserve':
            manager.reserve_interval(start, end)
            tree.delete_interval(start, end)
        elif op == 'release':
            manager.release_interval(start, end)
            tree.insert_interval(start, end)
        elif op == 'find':
            manager.find_interval(start, end - start)
            tree.find_interval(start, end - start)

    manager_intervals: List[Tuple[int, int]] = get_intervals(manager)
    tree_intervals: List[Tuple[int, int]] = get_tree_intervals(tree)
    
    print(f"IntervalManager execution time: {time_manager:.4f} seconds")
    print(f"EarliestIntervalTree execution time: {time_tree:.4f} seconds")
    # Check if the intervals cover the same ranges by merging adjacent intervals
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

    manager_merged = merge_intervals(manager_intervals)
    tree_merged = merge_intervals(tree_intervals)
    
    are_equivalent = manager_merged == tree_merged
    print(f"Data structures functionally equivalent: {are_equivalent}")
    if not are_equivalent:
        print("Effective coverage differs:")
        print("Manager coverage: ", end="")
        print("Tree coverage:    ", end="")
        print()
        # Find intervals that differ between the two data structures
        manager_diffs = [x for x in manager_merged if x not in tree_merged]
        tree_diffs = [x for x in tree_merged if x not in manager_merged]
        
        for i in range(max(len(manager_diffs), len(tree_diffs))):
            manager_interval = f"[{manager_diffs[i][0]}, {manager_diffs[i][1]}]*" if i < len(manager_diffs) else " " * 20
            tree_interval = f"[{tree_diffs[i][0]}, {tree_diffs[i][1]}]*" if i < len(tree_diffs) else ""
            print(f"{manager_interval:20} {tree_interval}")
    
    # Calculate speedup
    speedup: float = time_tree / time_manager if time_manager > 0 else float('inf')
    print(f"Speedup (Tree time / Manager time): {speedup:.2f}x")

if __name__ == "__main__":
    import sys
    run_benchmarks(random_seed=int(sys.argv[1]) if len(sys.argv) > 1 else None)