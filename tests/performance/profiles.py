"""Named, reproducible benchmark and load-qualification profiles."""

from __future__ import annotations

from dataclasses import dataclass

from tests.performance.harness import BenchmarkWorkload
from tests.performance.workload import (
    fragmented_workload,
    immutable_query_workload,
    lease_pool_workload,
    scheduling_workload,
)


@dataclass(frozen=True)
class BenchmarkProfile:
    """A fixed suite whose scale and methodology are part of its identity."""

    name: str
    description: str
    samples: int
    warmups: int
    processes: int
    payload_scale: int
    payload_operations: int
    sampled_workloads: tuple[BenchmarkWorkload, ...]
    qualification_workloads: tuple[BenchmarkWorkload, ...]


def benchmark_profile(name: str) -> BenchmarkProfile:
    """Build a deterministic profile without allocating unused large traces."""
    if name == "smoke":
        return BenchmarkProfile(
            name="smoke",
            description="PR-scale coverage of every canonical operation and stable backend",
            samples=20,
            warmups=1,
            processes=2,
            payload_scale=32,
            payload_operations=100,
            sampled_workloads=(
                fragmented_workload(interval_count=32, operation_count=200, seed=41),
                immutable_query_workload(
                    interval_count=128, queries_per_snapshot=1_100, seed=42
                ),
                scheduling_workload(cores=8, occupancy=0.50, jobs=300, seed=43),
                lease_pool_workload(shards=64, operation_count=200, seed=44),
            ),
            qualification_workloads=(
                immutable_query_workload(
                    interval_count=500, queries_per_snapshot=500, seed=45
                ),
                lease_pool_workload(shards=250, operation_count=500, seed=46),
            ),
        )
    if name == "standard":
        return BenchmarkProfile(
            name="standard",
            description="Weekly engineering benchmark plus medium-scale qualification",
            samples=20,
            warmups=2,
            processes=2,
            payload_scale=64,
            payload_operations=200,
            sampled_workloads=(
                fragmented_workload(interval_count=64, operation_count=500, seed=51),
                immutable_query_workload(
                    interval_count=128, queries_per_snapshot=1_100, seed=52
                ),
                scheduling_workload(cores=64, occupancy=0.75, jobs=1_000, seed=53),
                lease_pool_workload(shards=64, operation_count=500, seed=54),
            ),
            qualification_workloads=(
                immutable_query_workload(
                    interval_count=10_000, queries_per_snapshot=20, seed=55
                ),
                lease_pool_workload(shards=2_000, operation_count=200, seed=56),
                scheduling_workload(cores=64, occupancy=0.75, jobs=25_000, seed=57),
            ),
        )
    if name == "large":
        return BenchmarkProfile(
            name="large",
            description="Manual high-cardinality qualification under plausible production load",
            samples=20,
            warmups=2,
            processes=2,
            payload_scale=128,
            payload_operations=500,
            sampled_workloads=(
                fragmented_workload(interval_count=64, operation_count=500, seed=61),
                immutable_query_workload(
                    interval_count=128, queries_per_snapshot=1_100, seed=62
                ),
                scheduling_workload(cores=64, occupancy=0.75, jobs=1_000, seed=63),
                lease_pool_workload(shards=64, operation_count=500, seed=64),
            ),
            qualification_workloads=(
                immutable_query_workload(
                    interval_count=25_000,
                    queries_per_snapshot=20,
                    seed=65,
                ),
                lease_pool_workload(shards=5_000, operation_count=200, seed=66),
                scheduling_workload(cores=64, occupancy=0.75, jobs=50_000, seed=67),
            ),
        )
    raise ValueError(f"unknown benchmark profile: {name}")
