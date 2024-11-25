from typing import List, Tuple
import cProfile
from treemendous.basic.boundary import IntervalManager
from treemendous.basic.avl_earliest import EarliestIntervalTree
from tests.performance.benchmark import generate_operations

def profile_interval_manager(operations: List[Tuple[str, int, int]]) -> None:
    manager: IntervalManager = IntervalManager()
    manager.release_interval(0, 1_000_000)
    for op, start, end in operations:
        if op == 'reserve':
            manager.reserve_interval(start, end)
        elif op == 'release':
            manager.release_interval(start, end)
        elif op == 'find':
            manager.find_interval(start, end - start)

def profile_earliest_interval_tree(operations: List[Tuple[str, int, int]]) -> None:
    tree: EarliestIntervalTree = EarliestIntervalTree()
    tree.release_interval(0, 1_000_000)
    for op, start, end in operations:
        if op == 'reserve':
            tree.reserve_interval(start, end)
        elif op == 'release':
            tree.release_interval(start, end)
        elif op == 'find':
            tree.find_interval(start, end - start)

def run_profiling() -> None:
    num_operations: int = 10_000
    operations: List[Tuple[str, int, int]] = generate_operations(num_operations)

    cProfile.runctx('profile_interval_manager(operations)', globals(), locals(), 'interval_manager.prof')
    cProfile.runctx('profile_earliest_interval_tree(operations)', globals(), locals(), 'earliest_interval_tree.prof')

if __name__ == "__main__":
    run_profiling()