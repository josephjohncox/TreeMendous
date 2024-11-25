from typing import List, Tuple, Callable
import time
import random
from treemendous.basic.boundary import IntervalManager
from treemendous.basic.avl_earliest import EarliestIntervalTree

def benchmark_interval_manager(operations: List[Tuple[str, int, int]]) -> float:
    manager: IntervalManager = IntervalManager()
    manager.release_interval(0, 10_000_000)
    start_time: float = time.time()
    for op, start, end in operations:
        if op == 'reserve':
            manager.reserve_interval(start, end)
        elif op == 'release':
            manager.release_interval(start, end)
        elif op == 'find':
            manager.find_interval(start, end - start)
    return time.time() - start_time

def benchmark_interval_tree(operations: List[Tuple[str, int, int]]) -> float:
    tree: EarliestIntervalTree = EarliestIntervalTree()
    tree.insert_interval(0, 10_000_000)
    start_time: float = time.time()
    for op, start, end in operations:
        if op == 'reserve':
            tree.delete_interval(start, end)
        elif op == 'release':
            tree.insert_interval(start, end)
        elif op == 'find':
            tree.find_interval(start, end - start)
    return time.time() - start_time

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
    return sorted([(start, end) for start, end in manager.intervals.items()])

def get_tree_intervals(tree: EarliestIntervalTree) -> List[Tuple[int, int]]:
    return sorted(tree.get_all_intervals())

def run_benchmarks(random_seed: int | None) -> None:
    if random_seed:
        random.seed(random_seed)
    num_operations: int = 10_000
    operations: List[Tuple[str, int, int]] = generate_operations(num_operations)

    # Create fresh instances for validation
    manager: IntervalManager = IntervalManager()
    tree: EarliestIntervalTree = EarliestIntervalTree()
    manager.release_interval(0, 10_000_000)
    tree.insert_interval(0, 10_000_000)

    time_manager: float = benchmark_interval_manager(operations)
    time_tree: float = benchmark_interval_tree(operations)

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

if __name__ == "__main__":
    import sys
    run_benchmarks(random_seed=int(sys.argv[1]) if len(sys.argv) > 1 else None)