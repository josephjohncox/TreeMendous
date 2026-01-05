#!/usr/bin/env python3
"""
Unified Workload Generator for Tree-Mendous Benchmarks

Provides consistent, reproducible operation sequences for all benchmarks
and profiling tools. Ensures fair comparisons across all implementations.
"""

import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable, Any, Iterable


@dataclass(frozen=True)
class WorkloadProfile:
    """Realistic workload profile for benchmark sizing"""
    name: str
    operation_mix: Dict[str, float]
    space_range: Tuple[int, int]
    size_buckets: List[Tuple[int, int, float]]
    start_distribution: str = "uniform"  # "uniform" | "diurnal"
    align: int = 1  # Align starts/lengths to this granularity
    data_factory: Optional[Callable[[int, str, int, int], Any]] = None


REALISTIC_PROFILES: Dict[str, WorkloadProfile] = {
    # Scheduling workloads (minutes in a day)
    "scheduler": WorkloadProfile(
        name="scheduler",
        operation_mix={"reserve": 0.45, "release": 0.35, "find": 0.20},
        space_range=(0, 24 * 60),  # 24 hours in minutes
        size_buckets=[
            (1, 15, 0.60),    # short tasks
            (15, 120, 0.30),  # medium tasks
            (120, 480, 0.10), # long tasks
        ],
        start_distribution="diurnal",
        align=1,
    ),
    # Memory allocator workloads (bytes)
    "allocator": WorkloadProfile(
        name="allocator",
        operation_mix={"reserve": 0.55, "release": 0.35, "find": 0.10},
        space_range=(0, 1_000_000_000),  # 1 GB address space
        size_buckets=[
            (4_096, 65_536, 0.70),    # 4KB - 64KB
            (65_536, 1_048_576, 0.20), # 64KB - 1MB
            (1_048_576, 64_000_000, 0.10), # 1MB - 64MB
        ],
        start_distribution="uniform",
        align=4_096,
    ),
    # Network reservation workloads (Mbps over a day)
    "network": WorkloadProfile(
        name="network",
        operation_mix={"reserve": 0.50, "release": 0.35, "find": 0.15},
        space_range=(0, 100_000),  # 100 Gbps in Mbps
        size_buckets=[
            (10, 100, 0.55),     # small flows
            (100, 1_000, 0.30),  # medium flows
            (1_000, 10_000, 0.15), # large flows
        ],
        start_distribution="uniform",
        align=10,
    ),
}


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


def iter_workload(operations: List[Tuple]) -> Iterable[Tuple[str, int, int, Any]]:
    """Yield (op, start, end, data) for both 3- and 4-tuples."""
    for item in operations:
        if len(item) == 3:
            op, start, end = item
            yield op, start, end, None
        else:
            op, start, end, data = item
            yield op, start, end, data


def _sample_size(size_buckets: List[Tuple[int, int, float]], max_length: int, align: int) -> int:
    weights = [bucket[2] for bucket in size_buckets]
    bucket_min, bucket_max, _ = random.choices(size_buckets, weights=weights, k=1)[0]
    length = random.randint(bucket_min, bucket_max)
    if length > max_length:
        length = max_length
    if align > 1:
        if max_length < align:
            length = max_length
        else:
            length = max(align, (length // align) * align)
            length = min(length, max_length)
    return length


def _sample_start(space_min: int, space_max: int, length: int, distribution: str) -> int:
    max_start = max(space_min, space_max - length)
    if distribution == "diurnal":
        horizon = max_start - space_min
        if horizon <= 0:
            return space_min
        centers = [
            space_min + int(horizon * 0.25),
            space_min + int(horizon * 0.50),
            space_min + int(horizon * 0.75),
        ]
        weights = [0.4, 0.35, 0.25]
        for _ in range(10):
            center = random.choices(centers, weights=weights, k=1)[0]
            start = int(random.gauss(center, horizon * 0.08))
            if space_min <= start <= max_start:
                return start
    return random.randint(space_min, max_start)


def generate_realistic_workload(
    num_operations: int,
    profile: str = "scheduler",
    seed: int = 42,
    space_range: Optional[Tuple[int, int]] = None,
    operation_mix: Optional[Dict[str, float]] = None,
    include_data: bool = False,
    data_factory: Optional[Callable[[int, str, int, int], Any]] = None,
) -> List[Tuple]:
    """Generate workload using realistic sizing profiles."""
    if profile not in REALISTIC_PROFILES:
        raise ValueError(f"Unknown workload profile '{profile}' (available: {list(REALISTIC_PROFILES.keys())})")
    
    random.seed(seed)
    base_profile = REALISTIC_PROFILES[profile]
    space_min, space_max = space_range or base_profile.space_range
    
    op_mix = operation_mix or base_profile.operation_mix
    # Validate operation mix
    if abs(sum(op_mix.values()) - 1.0) > 0.01:
        raise ValueError(f"Operation mix must sum to 1.0, got {sum(op_mix.values())}")
    
    op_types = list(op_mix.keys())
    op_weights = list(op_mix.values())
    data_factory = data_factory or base_profile.data_factory
    
    max_length = max(1, space_max - space_min)
    operations = []
    
    for idx in range(num_operations):
        op_type = random.choices(op_types, weights=op_weights)[0]
        length = _sample_size(base_profile.size_buckets, max_length, base_profile.align)
        start = _sample_start(space_min, space_max, length, base_profile.start_distribution)
        end = start + length
        
        if include_data:
            payload = data_factory(idx, op_type, start, end) if data_factory else {
                "id": idx,
                "op": op_type,
                "size": length,
            }
            operations.append((op_type, start, end, payload))
        else:
            operations.append((op_type, start, end))
    
    return operations


def generate_standard_workload(num_operations: int = 10_000) -> List[Tuple[str, int, int]]:
    """
    Generate standard workload for typical benchmarks.
    
    Default configuration:
    - 40% reserve, 40% release, 20% find
    - Space: 0 to 1,000,000
    - Realistic scheduling-sized intervals (short/medium/long mix)
    """
    return generate_realistic_workload(
        num_operations=num_operations,
        profile="scheduler",
        space_range=(0, 999_900),
        operation_mix={'reserve': 0.4, 'release': 0.4, 'find': 0.2},
        seed=42,
        include_data=False
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


def _call_interval_op(impl, method_name: str, start: int, end: int, data: Any = None) -> None:
    method = getattr(impl, method_name)
    if data is None:
        try:
            method(start, end)
        except TypeError:
            method(start, end)
    else:
        try:
            method(start, end, data)
        except TypeError:
            method(start, end)


def execute_workload(impl, operations: List[Tuple], initial_space: Tuple[int, int] = (0, 1_000_000)) -> None:
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
    for op, start, end, data in iter_workload(operations):
        try:
            if op == 'reserve':
                _call_interval_op(impl, "reserve_interval", start, end, data)
            elif op == 'release':
                _call_interval_op(impl, "release_interval", start, end, data)
            elif op == 'find':
                impl.find_interval(start, end - start)
        except (ValueError, Exception):
            # Some operations may fail (no suitable interval, etc.)
            pass
