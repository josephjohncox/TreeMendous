"""Deterministic workload generators for correctness-checked benchmarks."""

from __future__ import annotations

import random
from math import floor

from tests.performance.harness import BenchmarkWorkload, Operation


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
    occupied = floor(horizon * occupancy)
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
