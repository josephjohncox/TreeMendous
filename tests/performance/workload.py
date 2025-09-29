#!/usr/bin/env python3
"""
Unified Workload Generator for Tree-Mendous Benchmarks

Provides consistent, reproducible operation sequences for all benchmarks
and profiling tools. Ensures fair comparisons across all implementations.
"""

import random
from typing import List, Tuple, Dict


def generate_workload(
    num_operations: int,
    seed: int = 42,
    operation_mix: Dict[str, float] = None,
    space_range: Tuple[int, int] = (0, 1_000_000),
    interval_size_range: Tuple[int, int] = (10, 100)
) -> List[Tuple[str, int, int]]:
    """
    Generate reproducible workload for benchmarking.
    
    Args:
        num_operations: Number of operations to generate
        seed: Random seed for reproducibility
        operation_mix: Distribution of operations {'reserve': 0.4, 'release': 0.4, 'find': 0.2}
        space_range: (min, max) for interval start positions
        interval_size_range: (min, max) for interval lengths
    
    Returns:
        List of (operation_type, start, end) tuples
    """
    random.seed(seed)
    
    if operation_mix is None:
        operation_mix = {'reserve': 0.4, 'release': 0.4, 'find': 0.2}
    
    # Validate operation mix
    if abs(sum(operation_mix.values()) - 1.0) > 0.01:
        raise ValueError(f"Operation mix must sum to 1.0, got {sum(operation_mix.values())}")
    
    op_types = list(operation_mix.keys())
    op_weights = list(operation_mix.values())
    
    operations = []
    space_min, space_max = space_range
    size_min, size_max = interval_size_range
    
    for _ in range(num_operations):
        op_type = random.choices(op_types, weights=op_weights)[0]
        start = random.randint(space_min, space_max - size_max)
        length = random.randint(size_min, size_max)
        end = start + length
        operations.append((op_type, start, end))
    
    return operations


def generate_standard_workload(num_operations: int = 10_000) -> List[Tuple[str, int, int]]:
    """
    Generate standard workload for typical benchmarks.
    
    Default configuration:
    - 40% reserve, 40% release, 20% find
    - Space: 0 to 1,000,000
    - Interval sizes: 10 to 100
    """
    return generate_workload(
        num_operations=num_operations,
        seed=42,
        operation_mix={'reserve': 0.4, 'release': 0.4, 'find': 0.2},
        space_range=(0, 999_900),
        interval_size_range=(10, 100)
    )


def generate_dense_workload(num_operations: int = 10_000, total_space: int = 1_000_000) -> List[Tuple[str, int, int]]:
    """
    Generate workload with many small intervals (high fragmentation).
    """
    return generate_workload(
        num_operations=num_operations,
        seed=42,
        operation_mix={'reserve': 0.5, 'release': 0.3, 'find': 0.2},
        space_range=(0, total_space - 1000),
        interval_size_range=(10, 100)
    )


def generate_sparse_workload(num_operations: int = 10_000, total_space: int = 10_000_000) -> List[Tuple[str, int, int]]:
    """
    Generate workload with few large intervals (low fragmentation).
    """
    return generate_workload(
        num_operations=num_operations,
        seed=42,
        operation_mix={'reserve': 0.4, 'release': 0.4, 'find': 0.2},
        space_range=(0, total_space - 100_000),
        interval_size_range=(1_000, 10_000)
    )


def generate_query_heavy_workload(num_operations: int = 10_000) -> List[Tuple[str, int, int]]:
    """
    Generate workload with emphasis on find operations.
    """
    return generate_workload(
        num_operations=num_operations,
        seed=42,
        operation_mix={'reserve': 0.2, 'release': 0.2, 'find': 0.6},
        space_range=(0, 999_900),
        interval_size_range=(10, 100)
    )


def execute_workload(impl, operations: List[Tuple[str, int, int]], initial_space: Tuple[int, int] = (0, 1_000_000)) -> None:
    """
    Execute a workload on any implementation (Python or C++).
    
    Works with any object that implements the standard protocol:
    - release_interval(start, end)
    - reserve_interval(start, end)
    - find_interval(start, length)
    
    Args:
        impl: Implementation instance (Python or C++)
        operations: List of (op_type, start, end) tuples
        initial_space: (start, end) for initial available space
    """
    # Initialize with available space
    impl.release_interval(*initial_space)
    
    # Execute operations
    for op, start, end in operations:
        try:
            if op == 'reserve':
                impl.reserve_interval(start, end)
            elif op == 'release':
                impl.release_interval(start, end)
            elif op == 'find':
                impl.find_interval(start, end - start)
        except (ValueError, Exception):
            # Some operations may fail (no suitable interval, etc.)
            pass
