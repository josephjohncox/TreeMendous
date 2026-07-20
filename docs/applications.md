# Concrete application engines

Tree-Mendous includes 50 reusable application engines in five production code
packages. These engines implement the named local semantics: document indexing
and search, regular-expression scanning, cumulative-capacity scheduling,
identity-preserving interval records, guarded allocation, fenced numeric leases,
and the other behaviors described in the scenario documents. They are not the
legacy application-shaped backend traces.

All 50 engines keep their working state in one Python process. A name such as
"distributed document search" describes the role the engine can play inside a
larger service, not a claim that this package supplies a distributed service.

## Creation and import boundary

The stable `treemendous` root continues to expose the one-dimensional `RangeSet`
API. It does not re-export application registry or engine symbols. Import the
registry explicitly when an application is selected by scenario ID:

```python
from treemendous.applications import create_application, get_scenario

spec = get_scenario("distributed-document-search")
search = create_application(
    spec.id,
    documents={10: "range indexes", 20: "unrelated text"},
    query="range",
)
hits = search.run(shard_size=1)
assert [hit.document_id for hit in hits] == [10]
```

`create_application` passes keyword arguments to the registered scenario
factory. Constructor arguments and returned types therefore differ by scenario.
Use the linked scenario document and example as the contract for that factory.
Code that depends on a concrete type should import it from its scenario module,
for example `treemendous.applications.partitioning.document_search`, rather
than from the `treemendous` root.

## Five application packages

| Family | Package | Engines | Shared implementation base |
| --- | --- | ---: | --- |
| Partitioning and work claiming | `treemendous.applications.partitioning` | 12 | owner-scoped claim ledgers and deterministic event streams |
| Scheduling and reservation | `treemendous.applications.scheduling` | 12 | cumulative-capacity reservation ledgers and deterministic placement |
| Identity-preserving overlap catalogs | `treemendous.applications.catalogs` | 10 | interval-record indexes, coverage projections, and range-lock tables |
| Allocation and capacity tracking | `treemendous.applications.allocation` | 8 | contiguous allocators, coverage tracking, and modular sequence kernels |
| Numeric resource leasing | `treemendous.applications.leasing` | 8 | lease pools, explicit clocks, lifecycle history, and local fence validation |

The reusable kernels live under `treemendous.applications._shared`. That module
is private implementation code. The concrete scenario modules add the domain
rules, validation, result types, diagnostics, and factory interfaces that
callers should use.

<!-- BEGIN GENERATED APPLICATION INDEX -->
## Partitioning and work claiming

Package: `treemendous.applications.partitioning`. Registered engines: 12.

| Application | Purpose | Example |
| --- | --- | --- |
| [Distributed document search](scenarios/partitioning/document_search.md) | workers claim document-ID bands and return abandoned bands for retry | [example](../examples/applications/partitioning/document_search.py) |
| [Distributed regular-expression scan](scenarios/partitioning/regex_scan.md) | workers claim byte-offset chunks while overlap checks model boundary halos | [example](../examples/applications/partitioning/regex_scan.py) |
| [Distributed genetic search](scenarios/partitioning/genetic_search.md) | workers claim population and candidate-ID bands with cancellation | [example](../examples/applications/partitioning/genetic_search.py) |
| [Distributed graph frontier search](scenarios/partitioning/graph_search.md) | workers claim vertex-ID frontier bands for parallel expansion | [example](../examples/applications/partitioning/graph_search.py) |
| [Distributed SAT search](scenarios/partitioning/sat_search.md) | workers claim ordinal ranges of assignment prefixes | [example](../examples/applications/partitioning/sat_search.py) |
| [Distributed fuzzing](scenarios/partitioning/fuzzing.md) | workers claim generated-input ordinal ranges and retry failed workers | [example](../examples/applications/partitioning/fuzzing.py) |
| [Distributed hyperparameter search](scenarios/partitioning/hyperparameter_search.md) | workers claim deterministic trial-ID ranges | [example](../examples/applications/partitioning/hyperparameter_search.py) |
| [Distributed log replay](scenarios/partitioning/log_replay.md) | consumers claim log-offset windows and return interrupted windows | [example](../examples/applications/partitioning/log_replay.py) |
| [Distributed build and test sharding](scenarios/partitioning/build_sharding.md) | runners claim ordered source-file or test-ID ranges | [example](../examples/applications/partitioning/build_sharding.py) |
| [Map-reduce input splitting](scenarios/partitioning/map_reduce.md) | workers claim byte and record ranges from partitioned inputs | [example](../examples/applications/partitioning/map_reduce.py) |
| [Distributed web crawl](scenarios/partitioning/web_crawl.md) | crawlers claim normalized URL-ID ranges | [example](../examples/applications/partitioning/web_crawl.py) |
| [Distributed search-index merge](scenarios/partitioning/index_merge.md) | workers claim term-ID and posting-list bands during index merges | [example](../examples/applications/partitioning/index_merge.py) |

## Scheduling and reservation

Package: `treemendous.applications.scheduling`. Registered engines: 12.

| Application | Purpose | Example |
| --- | --- | --- |
| [Distributed cluster scheduling](scenarios/scheduling/distributed-cluster-scheduling.md) | jobs reserve bounded compute-lane windows with deadlines and cancellation | [example](../examples/applications/scheduling/cluster.py) |
| [GPU stream scheduling](scenarios/scheduling/gpu-stream-scheduling.md) | kernels reserve stream-time windows under occupancy pressure | [example](../examples/applications/scheduling/gpu_streams.py) |
| [Render-farm frame scheduling](scenarios/scheduling/render-farm-frames.md) | workers reserve contiguous frame ranges and return cancelled renders | [example](../examples/applications/scheduling/render_farm.py) |
| [CI runner reservations](scenarios/scheduling/ci-runner-reservations.md) | jobs reserve runner-time windows with release times and deadlines | [example](../examples/applications/scheduling/ci_runners.py) |
| [Meeting-room booking](scenarios/scheduling/meeting-room-booking.md) | rooms expose available time ranges for bounded reservations | [example](../examples/applications/scheduling/meeting_rooms.py) |
| [Airline gate assignment](scenarios/scheduling/airline-gate-assignment.md) | flights reserve gate-time windows with turnaround constraints | [example](../examples/applications/scheduling/airline_gates.py) |
| [Operating-room booking](scenarios/scheduling/operating-room-booking.md) | procedures reserve room-time ranges and cancellations restore capacity | [example](../examples/applications/scheduling/operating_rooms.py) |
| [Laboratory equipment booking](scenarios/scheduling/laboratory-equipment-booking.md) | experiments reserve instrument-time windows | [example](../examples/applications/scheduling/lab_instruments.py) |
| [Fleet charging windows](scenarios/scheduling/fleet-charging-windows.md) | vehicles reserve charger-time ranges under capacity pressure | [example](../examples/applications/scheduling/fleet_charging.py) |
| [Radio spectrum timeslots](scenarios/scheduling/radio-spectrum-timeslots.md) | transmitters reserve channel-time windows | [example](../examples/applications/scheduling/radio_spectrum.py) |
| [Warehouse dock appointments](scenarios/scheduling/warehouse-dock-appointments.md) | carriers reserve dock-time windows and cancel stale appointments | [example](../examples/applications/scheduling/warehouse_docks.py) |
| [Maintenance-window planning](scenarios/scheduling/maintenance-window-planning.md) | services reserve maintenance windows within allowed periods | [example](../examples/applications/scheduling/maintenance.py) |

## Identity-preserving overlap catalogs

Package: `treemendous.applications.catalogs`. Registered engines: 10.

| Application | Purpose | Example |
| --- | --- | --- |
| [Genomic annotation overlap](scenarios/catalogs/genomic-annotation-overlap.md) | unioned gene, exon, variant, and read coverage is queried by overlap | [example](../examples/applications/catalogs/genomic_annotation_overlap.py) |
| [Source diagnostic ranges](scenarios/catalogs/source-diagnostic-ranges.md) | unioned diagnostic regions are queried by byte or token span | [example](../examples/applications/catalogs/source_diagnostic_ranges.py) |
| [Filesystem byte-range locks](scenarios/catalogs/filesystem-byte-locks.md) | lock requests query existing byte-range intersections | [example](../examples/applications/catalogs/filesystem_byte_locks.py) |
| [Database key-range locks](scenarios/catalogs/database-key-range-locks.md) | transactions query normalized integer key bands for conflicts | [example](../examples/applications/catalogs/database_key_range_locks.py) |
| [Packet sequence reassembly](scenarios/catalogs/packet-sequence-reassembly.md) | received sequence-number ranges are queried for overlap and gaps | [example](../examples/applications/catalogs/packet_sequence_reassembly.py) |
| [Subtitle cue ranges](scenarios/catalogs/subtitle-cue-ranges.md) | time-coded cues are queried at playback positions and edit windows | [example](../examples/applications/catalogs/subtitle_cue_ranges.py) |
| [Video edit regions](scenarios/catalogs/video-edit-regions.md) | frame ranges are queried for cuts, effects, and render invalidation | [example](../examples/applications/catalogs/video_edit_regions.py) |
| [Time-series alert windows](scenarios/catalogs/timeseries-alert-windows.md) | alert and suppression windows are queried by timestamp range | [example](../examples/applications/catalogs/timeseries_alert_windows.py) |
| [Distributed trace spans](scenarios/catalogs/distributed-trace-spans.md) | normalized timestamp ranges locate concurrent trace activity | [example](../examples/applications/catalogs/distributed_trace_spans.py) |
| [Morton-code geospatial ranges](scenarios/catalogs/morton-geospatial-ranges.md) | one-dimensional Morton-code bands approximate spatial query regions | [example](../examples/applications/catalogs/morton_geospatial_ranges.py) |

## Allocation and capacity tracking

Package: `treemendous.applications.allocation`. Registered engines: 8.

| Application | Purpose | Example |
| --- | --- | --- |
| [Heap free-space allocation](scenarios/allocation/heap_free_space.md) | allocators split, merge, reserve, and release address ranges | [example](../examples/applications/allocation/heap_free_space.py) |
| [Disk block allocation](scenarios/allocation/disk_block_allocation.md) | filesystems reserve and release contiguous block ranges | [example](../examples/applications/allocation/disk_block_allocation.py) |
| [Virtual address-space management](scenarios/allocation/virtual_address_space.md) | mappings reserve and release page-aligned virtual address ranges | [example](../examples/applications/allocation/virtual_address_space.py) |
| [Database page allocation](scenarios/allocation/database_page_allocation.md) | storage engines allocate and recycle page-ID ranges | [example](../examples/applications/allocation/database_page_allocation.py) |
| [Object-store multipart ranges](scenarios/allocation/object_store_multipart_ranges.md) | uploads track completed, missing, and retried byte ranges | [example](../examples/applications/allocation/object_store_multipart_ranges.py) |
| [CDN byte-range cache](scenarios/allocation/cdn_byte_range_cache.md) | cache entries track resident and evicted object byte ranges | [example](../examples/applications/allocation/cdn_byte_range_cache.py) |
| [GPU memory arena](scenarios/allocation/gpu_memory_arena.md) | buffers reserve and release device-address ranges | [example](../examples/applications/allocation/gpu_memory_arena.py) |
| [Ring-buffer sequence capacity](scenarios/allocation/ring_buffer_sequences.md) | producers reserve sequence-number ranges and consumers release them | [example](../examples/applications/allocation/ring_buffer_sequences.py) |

## Numeric resource leasing

Package: `treemendous.applications.leasing`. Registered engines: 8.

| Application | Purpose | Example |
| --- | --- | --- |
| [TCP and UDP port leases](scenarios/leasing/tcp_udp_port_leases.md) | services claim contiguous port ranges and return expired leases | [example](../examples/applications/leasing/tcp_udp_port_leases.py) |
| [Numeric IP address pools](scenarios/leasing/numeric_ip_address_pools.md) | address managers claim integer-encoded address ranges | [example](../examples/applications/leasing/numeric_ip_address_pools.py) |
| [Database ID pools](scenarios/leasing/database_id_pools.md) | writers claim contiguous identifier batches | [example](../examples/applications/leasing/database_id_pools.py) |
| [Software license seats](scenarios/leasing/software_license_seats.md) | clients claim seat-ID ranges and return expired sessions | [example](../examples/applications/leasing/software_license_seats.py) |
| [Warehouse bin ranges](scenarios/leasing/warehouse_bin_ranges.md) | inventory jobs claim contiguous normalized bin-ID ranges | [example](../examples/applications/leasing/warehouse_bin_ranges.py) |
| [Game-world region IDs](scenarios/leasing/game_world_region_ids.md) | servers claim region-ID bands and transfer ownership | [example](../examples/applications/leasing/game_world_region_ids.py) |
| [VLAN tag pools](scenarios/leasing/vlan_tag_pools.md) | network controllers lease contiguous VLAN identifier ranges | [example](../examples/applications/leasing/vlan_tag_pools.py) |
| [Phone extension pools](scenarios/leasing/phone_extension_pools.md) | provisioning systems claim extension-number ranges | [example](../examples/applications/leasing/phone_extension_pools.py) |

<!-- END GENERATED APPLICATION INDEX -->

## Guarantees and limits

The exact guarantee is scenario-specific. The common design is deterministic,
validated state transition logic over integer half-open coordinates. Depending
on the engine, this includes stable record identity, owner checks, retry
idempotency, non-mutating rejection, lifecycle history, immutable snapshots,
restorable checkpoints, and monotonically increasing local fencing tokens.
The linked documentation states which of these properties each engine provides.

The shared kernels lock their own in-process transitions. This does not make an
entire engine safe for arbitrary concurrent calls; some engines have additional
process-local state outside a kernel lock. Follow the concurrency boundary in
the scenario document.

The application packages do not provide:

- network transport, worker discovery, heartbeats, leader election, or
  distributed consensus;
- durable persistence, write-ahead logging, crash recovery, or exactly-once
  execution;
- an external authorization system or proof that a holder stopped using a
  released or expired resource;
- a durable fencing-token check at the protected downstream resource;
- globally optimal scheduling, bin packing, fairness, search quality, or
  convergence guarantees;
- a supported deployment-size envelope or performance threshold.

A snapshot is an observation and a checkpoint is serializable state. Neither is
durable until the caller stores it and controls writer handoff. Distributed
users must also enforce idempotent result commits and reject stale fencing
tokens at durable boundaries.

## Evidence and benchmarks

Every completed manifest row links to its concrete engine, executable example,
independent oracle, scenario benchmark, and scenario documentation in the
[implementation status table](use-cases.md#application-implementation-status).
The [benchmark guide](benchmarking.md) explains why the concrete application
suite and the legacy generic backend trace suite provide different evidence.
