"""Application-shaped interval workload matrix.

These scenarios test the range-management layer used by each application. They
intentionally do not claim that Tree-Mendous implements regex engines,
consensus, search algorithms, schedulers, or genetic algorithms themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from tests.performance.harness import BenchmarkWorkload, qualify_backends
from tests.performance.workload import (
    fragmented_workload,
    lease_pool_workload,
    overlap_query_workload,
    scheduling_workload,
)


@dataclass(frozen=True)
class ApplicationSpec:
    """One concrete use of integer half-open range management."""

    id: str
    title: str
    category: str
    family: str
    description: str


APPLICATION_SPECS = (
    # Distributed partition claiming and work stealing.
    ApplicationSpec(
        "distributed-document-search",
        "Distributed document search",
        "distributed_partition",
        "partition",
        "workers claim document-ID bands and return abandoned bands for retry",
    ),
    ApplicationSpec(
        "distributed-regex-scan",
        "Distributed regular-expression scan",
        "distributed_partition",
        "partition",
        "workers claim byte-offset chunks while overlap checks model boundary halos",
    ),
    ApplicationSpec(
        "distributed-genetic-search",
        "Distributed genetic search",
        "distributed_partition",
        "partition",
        "workers claim population and candidate-ID bands with cancellation",
    ),
    ApplicationSpec(
        "distributed-graph-search",
        "Distributed graph frontier search",
        "distributed_partition",
        "partition",
        "workers claim vertex-ID frontier bands for parallel expansion",
    ),
    ApplicationSpec(
        "distributed-sat-search",
        "Distributed SAT search",
        "distributed_partition",
        "partition",
        "workers claim ordinal ranges of assignment prefixes",
    ),
    ApplicationSpec(
        "distributed-fuzzing",
        "Distributed fuzzing",
        "distributed_partition",
        "partition",
        "workers claim generated-input ordinal ranges and retry failed workers",
    ),
    ApplicationSpec(
        "distributed-hyperparameter-search",
        "Distributed hyperparameter search",
        "distributed_partition",
        "partition",
        "workers claim deterministic trial-ID ranges",
    ),
    ApplicationSpec(
        "distributed-log-replay",
        "Distributed log replay",
        "distributed_partition",
        "partition",
        "consumers claim log-offset windows and return interrupted windows",
    ),
    ApplicationSpec(
        "distributed-build-sharding",
        "Distributed build and test sharding",
        "distributed_partition",
        "partition",
        "runners claim ordered source-file or test-ID ranges",
    ),
    ApplicationSpec(
        "map-reduce-input-splits",
        "Map-reduce input splitting",
        "distributed_partition",
        "partition",
        "workers claim byte and record ranges from partitioned inputs",
    ),
    ApplicationSpec(
        "distributed-web-crawl",
        "Distributed web crawl",
        "distributed_partition",
        "partition",
        "crawlers claim normalized URL-ID ranges",
    ),
    ApplicationSpec(
        "distributed-index-merge",
        "Distributed search-index merge",
        "distributed_partition",
        "partition",
        "workers claim term-ID and posting-list bands during index merges",
    ),
    # Scheduling and reservation.
    ApplicationSpec(
        "distributed-cluster-scheduling",
        "Distributed cluster scheduling",
        "scheduling_reservation",
        "scheduling",
        "jobs reserve bounded compute-lane windows with deadlines and cancellation",
    ),
    ApplicationSpec(
        "gpu-stream-scheduling",
        "GPU stream scheduling",
        "scheduling_reservation",
        "scheduling",
        "kernels reserve stream-time windows under occupancy pressure",
    ),
    ApplicationSpec(
        "render-farm-frames",
        "Render-farm frame scheduling",
        "scheduling_reservation",
        "scheduling",
        "workers reserve contiguous frame ranges and return cancelled renders",
    ),
    ApplicationSpec(
        "ci-runner-reservations",
        "CI runner reservations",
        "scheduling_reservation",
        "scheduling",
        "jobs reserve runner-time windows with release times and deadlines",
    ),
    ApplicationSpec(
        "meeting-room-booking",
        "Meeting-room booking",
        "scheduling_reservation",
        "scheduling",
        "rooms expose available time ranges for bounded reservations",
    ),
    ApplicationSpec(
        "airline-gate-assignment",
        "Airline gate assignment",
        "scheduling_reservation",
        "scheduling",
        "flights reserve gate-time windows with turnaround constraints",
    ),
    ApplicationSpec(
        "operating-room-booking",
        "Operating-room booking",
        "scheduling_reservation",
        "scheduling",
        "procedures reserve room-time ranges and cancellations restore capacity",
    ),
    ApplicationSpec(
        "laboratory-equipment-booking",
        "Laboratory equipment booking",
        "scheduling_reservation",
        "scheduling",
        "experiments reserve instrument-time windows",
    ),
    ApplicationSpec(
        "fleet-charging-windows",
        "Fleet charging windows",
        "scheduling_reservation",
        "scheduling",
        "vehicles reserve charger-time ranges under capacity pressure",
    ),
    ApplicationSpec(
        "radio-spectrum-timeslots",
        "Radio spectrum timeslots",
        "scheduling_reservation",
        "scheduling",
        "transmitters reserve channel-time windows",
    ),
    ApplicationSpec(
        "warehouse-dock-appointments",
        "Warehouse dock appointments",
        "scheduling_reservation",
        "scheduling",
        "carriers reserve dock-time windows and cancel stale appointments",
    ),
    ApplicationSpec(
        "maintenance-window-planning",
        "Maintenance-window planning",
        "scheduling_reservation",
        "scheduling",
        "services reserve maintenance windows within allowed periods",
    ),
    # Immutable overlap and annotation catalogs.
    ApplicationSpec(
        "genomic-annotation-overlap",
        "Genomic annotation overlap",
        "overlap_catalog",
        "catalog",
        "gene, exon, variant, and read spans are queried by coordinate overlap",
    ),
    ApplicationSpec(
        "source-diagnostic-ranges",
        "Source diagnostic ranges",
        "overlap_catalog",
        "catalog",
        "compiler and editor diagnostics are queried by byte or token span",
    ),
    ApplicationSpec(
        "filesystem-byte-locks",
        "Filesystem byte-range locks",
        "overlap_catalog",
        "catalog",
        "lock requests query existing byte-range intersections",
    ),
    ApplicationSpec(
        "database-key-range-locks",
        "Database key-range locks",
        "overlap_catalog",
        "catalog",
        "transactions query normalized integer key bands for conflicts",
    ),
    ApplicationSpec(
        "packet-sequence-reassembly",
        "Packet sequence reassembly",
        "overlap_catalog",
        "catalog",
        "received sequence-number ranges are queried for overlap and gaps",
    ),
    ApplicationSpec(
        "subtitle-cue-ranges",
        "Subtitle cue ranges",
        "overlap_catalog",
        "catalog",
        "time-coded cues are queried at playback positions and edit windows",
    ),
    ApplicationSpec(
        "video-edit-regions",
        "Video edit regions",
        "overlap_catalog",
        "catalog",
        "frame ranges are queried for cuts, effects, and render invalidation",
    ),
    ApplicationSpec(
        "timeseries-alert-windows",
        "Time-series alert windows",
        "overlap_catalog",
        "catalog",
        "alert and suppression windows are queried by timestamp range",
    ),
    ApplicationSpec(
        "distributed-trace-spans",
        "Distributed trace spans",
        "overlap_catalog",
        "catalog",
        "normalized timestamp ranges locate concurrent trace activity",
    ),
    ApplicationSpec(
        "morton-geospatial-ranges",
        "Morton-code geospatial ranges",
        "overlap_catalog",
        "catalog",
        "one-dimensional Morton-code bands approximate spatial query regions",
    ),
    # Allocation and free-space churn.
    ApplicationSpec(
        "heap-free-space",
        "Heap free-space allocation",
        "allocation_churn",
        "allocator",
        "allocators split, merge, reserve, and release address ranges",
    ),
    ApplicationSpec(
        "disk-block-allocation",
        "Disk block allocation",
        "allocation_churn",
        "allocator",
        "filesystems reserve and release contiguous block ranges",
    ),
    ApplicationSpec(
        "virtual-address-space",
        "Virtual address-space management",
        "allocation_churn",
        "allocator",
        "mappings reserve and release page-aligned virtual address ranges",
    ),
    ApplicationSpec(
        "database-page-allocation",
        "Database page allocation",
        "allocation_churn",
        "allocator",
        "storage engines allocate and recycle page-ID ranges",
    ),
    ApplicationSpec(
        "object-store-multipart-ranges",
        "Object-store multipart ranges",
        "allocation_churn",
        "allocator",
        "uploads track completed, missing, and retried byte ranges",
    ),
    ApplicationSpec(
        "cdn-byte-range-cache",
        "CDN byte-range cache",
        "allocation_churn",
        "allocator",
        "cache entries track resident and evicted object byte ranges",
    ),
    ApplicationSpec(
        "gpu-memory-arena",
        "GPU memory arena",
        "allocation_churn",
        "allocator",
        "buffers reserve and release device-address ranges",
    ),
    ApplicationSpec(
        "ring-buffer-sequences",
        "Ring-buffer sequence capacity",
        "allocation_churn",
        "allocator",
        "producers reserve sequence-number ranges and consumers release them",
    ),
    # Numeric resource leasing.
    ApplicationSpec(
        "tcp-udp-port-leases",
        "TCP and UDP port leases",
        "resource_leasing",
        "lease",
        "services claim contiguous port ranges and return expired leases",
    ),
    ApplicationSpec(
        "numeric-ip-address-pools",
        "Numeric IP address pools",
        "resource_leasing",
        "lease",
        "address managers claim integer-encoded address ranges",
    ),
    ApplicationSpec(
        "database-id-pools",
        "Database ID pools",
        "resource_leasing",
        "lease",
        "writers claim contiguous identifier batches",
    ),
    ApplicationSpec(
        "software-license-seats",
        "Software license seats",
        "resource_leasing",
        "lease",
        "clients claim seat-ID ranges and return expired sessions",
    ),
    ApplicationSpec(
        "warehouse-bin-ranges",
        "Warehouse bin ranges",
        "resource_leasing",
        "lease",
        "inventory jobs claim contiguous normalized bin-ID ranges",
    ),
    ApplicationSpec(
        "game-world-region-ids",
        "Game-world region IDs",
        "resource_leasing",
        "lease",
        "servers claim region-ID bands and transfer ownership",
    ),
    ApplicationSpec(
        "vlan-tag-pools",
        "VLAN tag pools",
        "resource_leasing",
        "lease",
        "network controllers lease contiguous VLAN identifier ranges",
    ),
    ApplicationSpec(
        "phone-extension-pools",
        "Phone extension pools",
        "resource_leasing",
        "lease",
        "provisioning systems claim extension-number ranges",
    ),
)


def _workload_for(
    spec: ApplicationSpec, *, scale: int, operations: int, seed: int
) -> BenchmarkWorkload:
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
        ),
    )


def application_scenarios(
    *, scale: int, operations: int
) -> tuple[tuple[ApplicationSpec, BenchmarkWorkload], ...]:
    """Build every application scenario at one bounded deterministic scale."""
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
    """Require every scenario to agree with the oracle on every stable backend."""
    reports: list[dict[str, Any]] = []
    for spec, workload in application_scenarios(scale=scale, operations=operations):
        report = qualify_backends(backend_ids, workload)
        report["application"] = {
            "id": spec.id,
            "title": spec.title,
            "category": spec.category,
            "family": spec.family,
            "description": spec.description,
        }
        reports.append(report)
    return reports
