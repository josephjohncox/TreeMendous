"""Deterministic, domain-shaped workloads for validated benchmarks."""

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
    """Create allocator churn over an intentionally fragmented free-space map."""
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
        "fragmented-allocator-churn",
        ((0, extent),),
        setup,
        tuple(operations),
        extent,
        (
            ("plausible_use", "free-space allocator with release/reserve churn"),
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
    """Query a large immutable catalog of disjoint available ranges.

    Periodic overlap, snapshot, and statistics reads exercise the rest of the
    canonical read surface under the same fragmented state.
    """
    if interval_count <= 0 or queries_per_snapshot <= 0:
        raise ValueError("interval and query counts must be positive")
    domain = tuple((index * 4, index * 4 + 2) for index in range(interval_count))
    extent = domain[-1][1]
    rng = random.Random(seed)
    operations: list[Operation] = []
    for index in range(queries_per_snapshot):
        interval_index = rng.randrange(interval_count)
        interval_start, interval_end = domain[interval_index]
        coordinate = rng.randrange(extent)
        if index == queries_per_snapshot - 1 or (index and index % 997 == 0):
            operations.append(Operation("snapshot"))
        elif index == queries_per_snapshot - 2 or (index and index % 499 == 0):
            operations.append(Operation("stats"))
        elif index == queries_per_snapshot - 3 or (index and index % 101 == 0):
            operations.append(
                Operation(
                    "overlaps",
                    start=interval_start,
                    end=interval_end,
                )
            )
        else:
            operations.append(
                Operation(
                    "first_fit",
                    length=rng.randint(1, 2),
                    not_before=coordinate,
                )
            )
    return BenchmarkWorkload(
        "immutable-capacity-catalog",
        domain,
        (),
        tuple(operations),
        extent,
        (
            ("plausible_use", "read-mostly capacity and availability lookup"),
            ("updates", "none during timed phase"),
            ("queries_per_snapshot", str(queries_per_snapshot)),
            ("read_surface", "first_fit, overlaps, snapshot, stats"),
        ),
        initially_available=True,
    )


def scheduling_workload(
    *,
    cores: int,
    occupancy: float,
    jobs: int = 500,
    seed: int = 42,
) -> BenchmarkWorkload:
    """Generate constrained allocation/cancellation work for resource lanes.

    Each core is represented by a disjoint managed span. Requests have release
    coordinates, exclusive deadlines, and short/medium/long durations. The
    canonical ``allocate`` call is the measured find-plus-atomic-reserve path.
    """
    if cores not in {1, 8, 64}:
        raise ValueError("cores must be one of 1, 8, or 64")
    if not 0.0 <= occupancy < 1.0:
        raise ValueError("occupancy must satisfy 0 <= occupancy < 1")
    if jobs <= 0:
        raise ValueError("jobs must be positive")
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
            ("plausible_use", "bounded compute-lane scheduling"),
            ("cores", str(cores)),
            ("occupancy", f"{occupancy:.2f}"),
            ("constraints", "release coordinate and exclusive deadline"),
            ("job_classes", "short=4,medium=16,long=64"),
            ("metrics", "allocate/cancel latency, success by class, Jain fairness"),
        ),
    )


def lease_pool_workload(
    *,
    shards: int = 1_000,
    slots_per_shard: int = 64,
    operation_count: int = 10_000,
    seed: int = 42,
) -> BenchmarkWorkload:
    """Model leases drawn from isolated port, ID, address, or seat pools."""
    if min(shards, slots_per_shard, operation_count) <= 0:
        raise ValueError("shards, slots, and operations must be positive")
    stride = slots_per_shard + 8
    domain = tuple(
        (shard * stride, shard * stride + slots_per_shard) for shard in range(shards)
    )
    extent = domain[-1][1]
    lease_sizes = {"single": 1, "block": 4, "burst": min(16, slots_per_shard)}
    rng = random.Random(seed)
    attempted: list[int] = []
    operations: list[Operation] = []
    for request_id in range(operation_count):
        if request_id and request_id % 1_009 == 0:
            operations.append(Operation("snapshot"))
            continue
        if request_id and request_id % 503 == 0:
            operations.append(Operation("stats"))
            continue
        if request_id and request_id % 211 == 0:
            shard = rng.randrange(shards)
            base = shard * stride
            start = base + rng.randrange(slots_per_shard)
            operations.append(
                Operation(
                    "overlaps", start=start, end=min(base + slots_per_shard, start + 4)
                )
            )
            continue
        if attempted and rng.random() < 0.20:
            operations.append(Operation("cancel", job_id=rng.choice(attempted)))
            continue
        lease_class = rng.choices(tuple(lease_sizes), weights=(0.70, 0.23, 0.07), k=1)[
            0
        ]
        length = lease_sizes[lease_class]
        shard = rng.randrange(shards)
        base = shard * stride
        not_before = base + rng.randint(0, slots_per_shard - length)
        operations.append(
            Operation(
                "allocate",
                length=length,
                not_before=not_before,
                not_after=base + slots_per_shard,
                job_id=request_id,
                job_class=lease_class,
            )
        )
        attempted.append(request_id)
    return BenchmarkWorkload(
        f"lease-pool-{shards}-shards",
        domain,
        (),
        tuple(operations),
        extent,
        (
            ("plausible_use", "sharded port, ID, address, or seat leasing"),
            ("shards", str(shards)),
            ("slots_per_shard", str(slots_per_shard)),
            ("lease_mix", "single=70%,block=23%,burst=7%"),
            ("cancellation_attempts", "20% with realistic duplicate/stale cancels"),
            ("read_checkpoints", "overlaps, snapshot, stats"),
        ),
        initially_available=True,
    )
