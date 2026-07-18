"""Deterministic workload generators for correctness-checked benchmarks."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from tests.performance.harness import BenchmarkWorkload, Operation


@dataclass(frozen=True)
class WorkloadProfile:
    """Legacy tuple-workload sizing profile."""

    name: str
    operation_mix: tuple[tuple[str, float], ...]
    space_range: tuple[int, int]
    size_range: tuple[int, int]


REALISTIC_PROFILES = {
    "scheduler": WorkloadProfile(
        "scheduler",
        (("reserve", 0.45), ("release", 0.35), ("find", 0.20)),
        (0, 24 * 60),
        (1, 120),
    ),
    "allocator": WorkloadProfile(
        "allocator",
        (("reserve", 0.55), ("release", 0.35), ("find", 0.10)),
        (0, 1_000_000),
        (4, 4_096),
    ),
    "network": WorkloadProfile(
        "network",
        (("reserve", 0.50), ("release", 0.35), ("find", 0.15)),
        (0, 100_000),
        (10, 1_000),
    ),
}


def generate_workload(
    num_operations: int,
    seed: int = 42,
    operation_mix: dict[str, float] | None = None,
    space_range: tuple[int, int] = (0, 1_000_000),
    interval_size_range: tuple[int, int] = (10, 100),
) -> list[tuple[str, int, int]]:
    """Generate a reproducible legacy tuple trace without global RNG state."""
    if num_operations < 0:
        raise ValueError("num_operations cannot be negative")
    mix = operation_mix or {"reserve": 0.4, "release": 0.4, "find": 0.2}
    if abs(sum(mix.values()) - 1.0) > 1e-9:
        raise ValueError("operation mix must sum to 1.0")
    minimum, maximum = space_range
    size_minimum, size_maximum = interval_size_range
    if minimum >= maximum or size_minimum <= 0 or size_minimum > size_maximum:
        raise ValueError("invalid coordinate or interval-size range")
    if maximum - minimum < size_maximum:
        raise ValueError("space is smaller than the maximum interval")

    rng = random.Random(seed)
    kinds = list(mix)
    weights = list(mix.values())
    result = []
    for _ in range(num_operations):
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        length = rng.randint(size_minimum, size_maximum)
        start = rng.randint(minimum, maximum - length)
        result.append((kind, start, start + length))
    return result


def generate_realistic_workload(
    num_operations: int,
    profile: str = "scheduler",
    seed: int = 42,
    space_range: tuple[int, int] | None = None,
    operation_mix: dict[str, float] | None = None,
    include_data: bool = False,
    data_factory=None,
) -> list[tuple]:
    """Compatibility generator used by non-publishable profiling scripts."""
    selected = REALISTIC_PROFILES[profile]
    tuples = generate_workload(
        num_operations,
        seed,
        operation_mix or dict(selected.operation_mix),
        space_range or selected.space_range,
        selected.size_range,
    )
    if not include_data:
        return tuples
    factory = data_factory or (
        lambda index, kind, start, end: {
            "id": index,
            "op": kind,
            "size": end - start,
        }
    )
    return [
        (kind, start, end, factory(index, kind, start, end))
        for index, (kind, start, end) in enumerate(tuples)
    ]


def generate_standard_workload(
    num_operations: int = 10_000,
) -> list[tuple[str, int, int]]:
    """Generate the stable legacy tuple workload."""
    return generate_workload(num_operations, interval_size_range=(1, 120))


def iter_workload(
    operations: Iterable[tuple],
) -> Iterable[tuple[str, int, int, object | None]]:
    """Normalize legacy three- and four-field tuples."""
    for item in operations:
        if len(item) == 3:
            kind, start, end = item
            yield kind, start, end, None
        elif len(item) == 4:
            kind, start, end, data = item
            yield kind, start, end, data
        else:
            raise ValueError("workload tuples must contain three or four fields")


def execute_workload(
    implementation,
    operations: Iterable[tuple],
    initial_space: tuple[int, int] = (0, 1_000_000),
) -> tuple[tuple[int, int] | None, ...]:
    """Execute an ordered legacy trace without swallowing backend failures."""
    implementation.release_interval(*initial_space)
    queries = []
    for kind, start, end, data in iter_workload(operations):
        if kind == "reserve":
            implementation.reserve_interval(start, end)
        elif kind == "release":
            if data is None:
                implementation.release_interval(start, end)
            else:
                implementation.release_interval(start, end, data)
        elif kind == "find":
            found = implementation.find_interval(start, end - start)
            queries.append(
                None
                if found is None
                else (
                    (found.start, found.end)
                    if hasattr(found, "start")
                    else (found[0], found[1])
                )
            )
        else:
            raise ValueError(f"unknown operation kind: {kind}")
    return tuple(queries)


def fragmented_workload(
    *,
    interval_count: int = 64,
    operation_count: int = 500,
    seed: int = 42,
) -> BenchmarkWorkload:
    """Create a mutation-heavy trace with an exact initial interval count."""
    if interval_count <= 0 or operation_count <= 0:
        raise ValueError("interval and operation counts must be positive")
    extent = interval_count * 4
    setup = tuple(
        Operation("add", start=index * 4, end=index * 4 + 2)
        for index in range(interval_count)
    )
    rng = random.Random(seed)
    operations: list[Operation] = []
    for index in range(operation_count):
        length = rng.randint(1, min(12, extent))
        start = rng.randint(0, extent - length)
        selector = rng.random()
        if index == operation_count - 1 or (index and index % 97 == 0):
            operations.append(
                Operation("add", start=start, end=start, expected_error="ValueError")
            )
        elif selector < 0.35:
            operations.append(Operation("discard", start=start, end=start + length))
        elif selector < 0.70:
            operations.append(Operation("add", start=start, end=start + length))
        elif selector < 0.85:
            operations.append(Operation("first_fit", length=length, not_before=start))
        else:
            operations.append(Operation("allocate", length=length, not_before=start))
    return BenchmarkWorkload(
        "mutation-heavy",
        ((0, extent),),
        setup,
        tuple(operations),
        extent,
        (
            ("update_query_ratio", "70:30"),
            ("fragmentation", "alternating equal free/reserved spans"),
            ("fit_positions", "uniform not_before with impossible fits included"),
        ),
    )


def immutable_query_workload(
    *,
    interval_count: int = 64,
    queries_per_snapshot: int = 500,
    seed: int = 42,
) -> BenchmarkWorkload:
    """Create a fragmented immutable snapshot followed only by fit queries."""
    base = fragmented_workload(
        interval_count=interval_count, operation_count=1, seed=seed
    )
    rng = random.Random(seed)
    operations = tuple(
        Operation(
            "first_fit",
            length=rng.randint(1, 5),
            not_before=rng.randint(0, base.coordinate_extent - 1),
        )
        for _ in range(queries_per_snapshot)
    )
    return BenchmarkWorkload(
        "immutable-snapshot-batched-query",
        base.domain,
        base.setup,
        operations,
        base.coordinate_extent,
        (
            ("updates", "none during timed phase"),
            ("queries_per_snapshot", str(queries_per_snapshot)),
            ("upload_policy", "one immutable snapshot per query batch"),
        ),
    )


def scheduling_workload(
    *,
    cores: int,
    occupancy: float,
    jobs: int = 500,
    seed: int = 42,
) -> BenchmarkWorkload:
    """Generate domain-neutral constrained allocation/cancellation work.

    Each core is represented by a disjoint managed span. Requests have release
    coordinates, exclusive deadlines, and short/medium/long durations. The
    canonical ``allocate`` call is the measured find-plus-atomic-reserve path.
    """
    if cores not in {1, 8, 64}:
        raise ValueError("cores must be one of 1, 8, or 64")
    if not 0.0 <= occupancy < 1.0:
        raise ValueError("occupancy must satisfy 0 <= occupancy < 1")
    horizon = 512
    stride = horizon + 16
    domain = tuple((core * stride, core * stride + horizon) for core in range(cores))
    occupied = int(horizon * occupancy)
    setup = tuple(
        Operation("add", start=base + occupied, end=base + horizon)
        for base, _ in domain
    )
    durations = {"short": 4, "medium": 16, "long": 64}
    rng = random.Random(seed)
    operations: list[Operation] = []
    attempted_jobs: list[int] = []
    for job_id in range(jobs):
        if attempted_jobs and rng.random() < 0.12:
            operations.append(Operation("cancel", job_id=rng.choice(attempted_jobs)))
            continue
        job_class = rng.choices(tuple(durations), weights=(0.55, 0.30, 0.15), k=1)[0]
        duration = durations[job_class]
        core = rng.randrange(cores)
        base = core * stride
        release = rng.randint(0, horizon - duration)
        slack = rng.choice((0, duration, duration * 3))
        deadline = min(horizon, release + duration + slack)
        operations.append(
            Operation(
                "allocate",
                length=duration,
                not_before=base + release,
                not_after=base + deadline,
                job_id=job_id,
                job_class=job_class,
            )
        )
        attempted_jobs.append(job_id)
    return BenchmarkWorkload(
        f"cpu-scheduling-{cores}-cores-{occupancy:.0%}-occupancy",
        domain,
        setup,
        tuple(operations),
        cores * horizon,
        (
            ("cores", str(cores)),
            ("occupancy", f"{occupancy:.2f}"),
            ("constraints", "release coordinate and exclusive deadline"),
            ("job_classes", "short=4,medium=16,long=64"),
            ("metrics", "allocate/cancel latency, success by class, Jain fairness"),
        ),
    )
