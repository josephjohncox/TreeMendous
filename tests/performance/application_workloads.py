"""Legacy application-shaped interval qualification matrix.

These benchmark-only traces exercise five generic range-operation families.
They are not application engines and never determine scenario completion status.
The authoritative immutable scenario metadata lives in
:mod:`treemendous.applications.registry`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from tests.performance.harness import BenchmarkWorkload, qualify_backends
from tests.performance.workload import (
    fragmented_workload,
    lease_pool_workload,
    overlap_query_workload,
    scheduling_workload,
)
from treemendous.applications import SCENARIO_SPECS, ScenarioSpec

ApplicationSpec = ScenarioSpec
APPLICATION_SPECS = SCENARIO_SPECS


def _workload_for(
    spec: ScenarioSpec, *, scale: int, operations: int, seed: int
) -> BenchmarkWorkload:
    """Map metadata to a generic trace; this is not an engine factory."""
    if spec.family == "partition":
        base = lease_pool_workload(
            shards=scale,
            slots_per_shard=256,
            operation_count=operations,
            seed=seed,
        )
    elif spec.family == "scheduling":
        base = scheduling_workload(
            cores=8 if scale < 32 else 64,
            occupancy=0.75,
            jobs=operations,
            seed=seed,
        )
    elif spec.family == "catalog":
        base = overlap_query_workload(
            interval_count=scale,
            query_count=operations,
            seed=seed,
        )
    elif spec.family == "allocator":
        base = fragmented_workload(
            interval_count=scale,
            operation_count=operations,
            seed=seed,
        )
    elif spec.family == "lease":
        base = lease_pool_workload(
            shards=scale,
            slots_per_shard=64,
            operation_count=operations,
            seed=seed,
        )
    else:
        raise ValueError(f"unknown application workload family: {spec.family}")
    return replace(
        base,
        name=f"application-{spec.id}",
        dimensions=base.dimensions
        + (
            ("application_id", spec.id),
            ("application", spec.title),
            ("category", spec.category),
            ("workload_family", spec.family),
            ("plausible_use", spec.description),
            ("implementation_status", spec.status.value),
        ),
    )


def application_scenarios(
    *, scale: int, operations: int
) -> tuple[tuple[ScenarioSpec, BenchmarkWorkload], ...]:
    """Build every legacy application-shaped trace at one bounded scale."""
    if min(scale, operations) <= 0:
        raise ValueError("application scale and operations must be positive")
    return tuple(
        (
            spec,
            _workload_for(
                spec,
                scale=scale,
                operations=operations,
                seed=10_000 + index,
            ),
        )
        for index, spec in enumerate(APPLICATION_SPECS)
    )


def qualify_application_scenarios(
    backend_ids: tuple[str, ...], *, scale: int, operations: int
) -> list[dict[str, Any]]:
    """Qualify generic traces without claiming application implementation."""
    reports: list[dict[str, Any]] = []
    for spec, workload in application_scenarios(scale=scale, operations=operations):
        report = qualify_backends(backend_ids, workload)
        report["application"] = {
            "id": spec.id,
            "title": spec.title,
            "category": spec.category,
            "family": spec.family,
            "description": spec.description,
            "implementation_status": spec.status.value,
            "evidence": "application-shaped generic trace; not an engine",
        }
        reports.append(report)
    return reports
