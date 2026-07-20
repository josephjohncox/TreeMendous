# Interval workload and use-case matrix

Tree-Mendous manages one-dimensional integer half-open ranges. It can represent
which coordinates are free, claimed, annotated, reserved, or overlapping. The
50 names below are application-shaped benchmark traces, not application
engines. At this revision, **0/50** are implemented as real reusable engines.
Tree-Mendous does not yet implement the surrounding search engine,
regular-expression engine, scheduler, genetic algorithm, database, network
controller, or distributed consensus protocol.

For distributed applications, Tree-Mendous can be the deterministic range-state
component inside one coordinator or a replicated state machine. The surrounding
system must still provide durable storage, fencing tokens, leases, consensus,
idempotency, worker liveness, and recovery. Sharing an in-memory `RangeSet`
between unrelated processes is not a distributed protocol.

<!-- BEGIN GENERATED SCENARIO STATUS -->
## Application implementation status

Current completion: **0/50** real engines.
A benchmark trace is not implementation evidence. An entry becomes
`COMPLETE` only when its engine, example, independent oracle, benchmark,
and scenario documentation are all registered.

| Scenario | Family | Category | Status | Engine | Example | Oracle | Benchmark | Docs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `distributed-document-search` | `partition` | `distributed_partition` | `PLANNED` | missing | missing | missing | missing | missing |
| `distributed-regex-scan` | `partition` | `distributed_partition` | `PLANNED` | missing | missing | missing | missing | missing |
| `distributed-genetic-search` | `partition` | `distributed_partition` | `PLANNED` | missing | missing | missing | missing | missing |
| `distributed-graph-search` | `partition` | `distributed_partition` | `PLANNED` | missing | missing | missing | missing | missing |
| `distributed-sat-search` | `partition` | `distributed_partition` | `PLANNED` | missing | missing | missing | missing | missing |
| `distributed-fuzzing` | `partition` | `distributed_partition` | `PLANNED` | missing | missing | missing | missing | missing |
| `distributed-hyperparameter-search` | `partition` | `distributed_partition` | `PLANNED` | missing | missing | missing | missing | missing |
| `distributed-log-replay` | `partition` | `distributed_partition` | `PLANNED` | missing | missing | missing | missing | missing |
| `distributed-build-sharding` | `partition` | `distributed_partition` | `PLANNED` | missing | missing | missing | missing | missing |
| `map-reduce-input-splits` | `partition` | `distributed_partition` | `PLANNED` | missing | missing | missing | missing | missing |
| `distributed-web-crawl` | `partition` | `distributed_partition` | `PLANNED` | missing | missing | missing | missing | missing |
| `distributed-index-merge` | `partition` | `distributed_partition` | `PLANNED` | missing | missing | missing | missing | missing |
| `distributed-cluster-scheduling` | `scheduling` | `scheduling_reservation` | `PLANNED` | missing | missing | missing | missing | missing |
| `gpu-stream-scheduling` | `scheduling` | `scheduling_reservation` | `PLANNED` | missing | missing | missing | missing | missing |
| `render-farm-frames` | `scheduling` | `scheduling_reservation` | `PLANNED` | missing | missing | missing | missing | missing |
| `ci-runner-reservations` | `scheduling` | `scheduling_reservation` | `PLANNED` | missing | missing | missing | missing | missing |
| `meeting-room-booking` | `scheduling` | `scheduling_reservation` | `PLANNED` | missing | missing | missing | missing | missing |
| `airline-gate-assignment` | `scheduling` | `scheduling_reservation` | `PLANNED` | missing | missing | missing | missing | missing |
| `operating-room-booking` | `scheduling` | `scheduling_reservation` | `PLANNED` | missing | missing | missing | missing | missing |
| `laboratory-equipment-booking` | `scheduling` | `scheduling_reservation` | `PLANNED` | missing | missing | missing | missing | missing |
| `fleet-charging-windows` | `scheduling` | `scheduling_reservation` | `PLANNED` | missing | missing | missing | missing | missing |
| `radio-spectrum-timeslots` | `scheduling` | `scheduling_reservation` | `PLANNED` | missing | missing | missing | missing | missing |
| `warehouse-dock-appointments` | `scheduling` | `scheduling_reservation` | `PLANNED` | missing | missing | missing | missing | missing |
| `maintenance-window-planning` | `scheduling` | `scheduling_reservation` | `PLANNED` | missing | missing | missing | missing | missing |
| `genomic-annotation-overlap` | `catalog` | `overlap_catalog` | `PLANNED` | missing | missing | missing | missing | missing |
| `source-diagnostic-ranges` | `catalog` | `overlap_catalog` | `PLANNED` | missing | missing | missing | missing | missing |
| `filesystem-byte-locks` | `catalog` | `overlap_catalog` | `PLANNED` | missing | missing | missing | missing | missing |
| `database-key-range-locks` | `catalog` | `overlap_catalog` | `PLANNED` | missing | missing | missing | missing | missing |
| `packet-sequence-reassembly` | `catalog` | `overlap_catalog` | `PLANNED` | missing | missing | missing | missing | missing |
| `subtitle-cue-ranges` | `catalog` | `overlap_catalog` | `PLANNED` | missing | missing | missing | missing | missing |
| `video-edit-regions` | `catalog` | `overlap_catalog` | `PLANNED` | missing | missing | missing | missing | missing |
| `timeseries-alert-windows` | `catalog` | `overlap_catalog` | `PLANNED` | missing | missing | missing | missing | missing |
| `distributed-trace-spans` | `catalog` | `overlap_catalog` | `PLANNED` | missing | missing | missing | missing | missing |
| `morton-geospatial-ranges` | `catalog` | `overlap_catalog` | `PLANNED` | missing | missing | missing | missing | missing |
| `heap-free-space` | `allocator` | `allocation_churn` | `PLANNED` | missing | missing | missing | missing | missing |
| `disk-block-allocation` | `allocator` | `allocation_churn` | `PLANNED` | missing | missing | missing | missing | missing |
| `virtual-address-space` | `allocator` | `allocation_churn` | `PLANNED` | missing | missing | missing | missing | missing |
| `database-page-allocation` | `allocator` | `allocation_churn` | `PLANNED` | missing | missing | missing | missing | missing |
| `object-store-multipart-ranges` | `allocator` | `allocation_churn` | `PLANNED` | missing | missing | missing | missing | missing |
| `cdn-byte-range-cache` | `allocator` | `allocation_churn` | `PLANNED` | missing | missing | missing | missing | missing |
| `gpu-memory-arena` | `allocator` | `allocation_churn` | `PLANNED` | missing | missing | missing | missing | missing |
| `ring-buffer-sequences` | `allocator` | `allocation_churn` | `PLANNED` | missing | missing | missing | missing | missing |
| `tcp-udp-port-leases` | `lease` | `resource_leasing` | `PLANNED` | missing | missing | missing | missing | missing |
| `numeric-ip-address-pools` | `lease` | `resource_leasing` | `PLANNED` | missing | missing | missing | missing | missing |
| `database-id-pools` | `lease` | `resource_leasing` | `PLANNED` | missing | missing | missing | missing | missing |
| `software-license-seats` | `lease` | `resource_leasing` | `PLANNED` | missing | missing | missing | missing | missing |
| `warehouse-bin-ranges` | `lease` | `resource_leasing` | `PLANNED` | missing | missing | missing | missing | missing |
| `game-world-region-ids` | `lease` | `resource_leasing` | `PLANNED` | missing | missing | missing | missing | missing |
| `vlan-tag-pools` | `lease` | `resource_leasing` | `PLANNED` | missing | missing | missing | missing | missing |
| `phone-extension-pools` | `lease` | `resource_leasing` | `PLANNED` | missing | missing | missing | missing | missing |
<!-- END GENERATED SCENARIO STATUS -->

The benchmark suite executes the following 50 application-shaped scenarios
against every stable backend. These are deliberately generic range-operation
qualifications mapped onto plausible names; they are not factories for the
named applications and cannot mark a manifest entry complete. Each trace is
accepted only when its complete state, query results, mutation accounting,
snapshots, statistics, and overlap observations match the independent range
oracle.

## Distributed partition claiming

These scenarios allocate bounded work ranges to workers, return cancelled work,
and check overlapping claims. The range layer partitions work; it does not
perform the work itself.

| Scenario | Range interpretation |
| --- | --- |
| `distributed-document-search` | Document-ID bands claimed by search workers |
| `distributed-regex-scan` | Byte-offset chunks claimed by regex workers; overlap checks represent boundary halos |
| `distributed-genetic-search` | Population and candidate-ID bands claimed by evaluation workers |
| `distributed-graph-search` | Vertex-ID frontier bands claimed for parallel expansion |
| `distributed-sat-search` | Ordinal ranges of assignment prefixes |
| `distributed-fuzzing` | Generated-input ordinal ranges with failed-worker retry |
| `distributed-hyperparameter-search` | Deterministic trial-ID ranges |
| `distributed-log-replay` | Log-offset windows assigned to consumers |
| `distributed-build-sharding` | Ordered source-file or test-ID ranges |
| `map-reduce-input-splits` | Input byte or record partitions |
| `distributed-web-crawl` | Normalized URL-ID bands |
| `distributed-index-merge` | Term-ID and posting-list bands |

A distributed regex scan usually needs a halo at chunk boundaries so a match
crossing two chunks is not lost. The benchmark models claim and overlap
geometry, but the caller must choose the halo width from the regex semantics and
must deduplicate matches from adjacent workers.

A distributed genetic search can use ranges for deterministic candidate IDs,
random seeds, or population slots. The benchmark validates allocation,
cancellation, and retry geometry; selection, crossover, mutation, fitness, and
random-number reproducibility remain application responsibilities.

## Scheduling and reservation

These scenarios use `allocate` with release coordinates and exclusive deadlines,
then restore cancelled ranges.

| Scenario | Range interpretation |
| --- | --- |
| `distributed-cluster-scheduling` | Compute-lane time windows |
| `gpu-stream-scheduling` | GPU stream-time windows |
| `render-farm-frames` | Contiguous frame ranges assigned to workers |
| `ci-runner-reservations` | Runner-time windows |
| `meeting-room-booking` | Room availability by time coordinate |
| `airline-gate-assignment` | Gate occupancy and turnaround windows |
| `operating-room-booking` | Procedure and room-time windows |
| `laboratory-equipment-booking` | Instrument-time windows |
| `fleet-charging-windows` | Charger-time windows |
| `radio-spectrum-timeslots` | Channel-time windows |
| `warehouse-dock-appointments` | Dock-time windows |
| `maintenance-window-planning` | Allowed service-maintenance windows |

Tree-Mendous performs deterministic range selection and atomic in-process
reservation. It does not solve global optimization, precedence constraints,
multi-resource matching, preemption policy, or distributed transaction commit.
Those layers can propose a candidate range and commit it through `RangeSet`.

## Immutable overlap and annotation catalogs

These scenarios are read-heavy catalogs with 80% overlap queries, fit queries,
and snapshot/statistics checkpoints. They validate unioned region geometry, not
retrieval of every source record; preserve record identities externally or in a
lawful payload model.

| Scenario | Range interpretation |
| --- | --- |
| `genomic-annotation-overlap` | Unions of gene, exon, variant, and read coverage |
| `source-diagnostic-ranges` | Unions of compiler and editor diagnostic regions |
| `filesystem-byte-locks` | File byte-range locks |
| `database-key-range-locks` | Normalized integer key bands |
| `packet-sequence-reassembly` | Received packet sequence-number ranges |
| `subtitle-cue-ranges` | Time-coded caption cues |
| `video-edit-regions` | Frame ranges for cuts, effects, and invalidation |
| `timeseries-alert-windows` | Alert and suppression timestamp windows |
| `distributed-trace-spans` | Normalized trace timestamp ranges |
| `morton-geospatial-ranges` | One-dimensional Morton-code approximations of spatial regions |

Coordinates must already be integers with a meaningful total order. Floating
timestamps should be converted to fixed units. Database strings need a stable
order-preserving integer encoding. Morton or Hilbert ranges remain approximate
when modeled through one-dimensional `RangeSet`; the separate experimental
`treemendous.multidimensional.BoxIndex` preserves exact axis-aligned boxes but
is not a completed application engine for this catalog.

## Allocation and free-space churn

These scenarios split, merge, reserve, release, and search fragmented capacity.

| Scenario | Range interpretation |
| --- | --- |
| `heap-free-space` | Heap address ranges |
| `disk-block-allocation` | Filesystem block ranges |
| `virtual-address-space` | Page-aligned virtual addresses |
| `database-page-allocation` | Storage-engine page IDs |
| `object-store-multipart-ranges` | Completed, missing, and retried object bytes |
| `cdn-byte-range-cache` | Resident and evicted object bytes |
| `gpu-memory-arena` | Device-memory address ranges |
| `ring-buffer-sequences` | Producer and consumer sequence numbers |

Alignment, page size, allocation strategy, compaction, eviction, and persistence
belong to the caller. A first fit in integer coordinates is not automatically a
correct aligned fit; normalize units or round candidates before committing.

## Numeric resource leasing

These scenarios allocate contiguous numeric resource blocks and restore expired
or cancelled leases.

| Scenario | Range interpretation |
| --- | --- |
| `tcp-udp-port-leases` | Port-number ranges |
| `numeric-ip-address-pools` | Integer-encoded IP address ranges |
| `database-id-pools` | Identifier batches |
| `software-license-seats` | Seat-ID ranges |
| `warehouse-bin-ranges` | Normalized bin-ID ranges |
| `game-world-region-ids` | Region ownership bands |
| `vlan-tag-pools` | VLAN identifier ranges |
| `phone-extension-pools` | Extension-number ranges |

Lease expiration needs an external clock and durable lease record. Tree-Mendous
can return an expired span to availability, but it does not prove that the old
holder stopped using the resource. Distributed users need fencing or generation
numbers.

## Payload-policy applications

The separate payload qualification covers three recurring application shapes:

- `UniformPayloadPolicy`: tenant, owner, class, or state labels that may merge
  only when equal;
- `JoinPayloadPolicy`: commutative/idempotent permission, ownership, feature, or
  coverage overlays;
- `OrderedPayloadPolicy`: coordinate/event-key ordered booking, edit, or rule
  composition.

A custom join or ordered combine function is caller code. It must obey the
policy's documented algebraic laws, terminate, avoid hidden mutation, and remain
deterministic across processes if results are replicated.

## What this matrix does not prove

No finite suite proves every possible use case. This matrix establishes that the
shared range operations work across five operation families, 50 labeled
application-shaped traces, all explicit payload policies, and every stable
backend. It does not establish:

- generic interval-multimap semantics that preserve every overlapping record or
  duplicate identity; `RangeSet` canonicalizes geometry and folds payloads;
- correctness for continuous, multidimensional, cyclic, unbounded, or partially
  ordered domains without an explicit finite integer encoding;
- thread/process safety beyond the documented in-process `RangeSet` lock;
- persistence, crash recovery, distributed consensus, or exactly-once work;
- application-specific optimality, fairness policy, regex semantics, genetic
  convergence, or search quality;
- acceptable performance at arbitrary cardinality.

Use the smoke profile for semantic coverage, the standard profile for weekly
engineering evidence, and the sharded large profile for bounded high-cardinality
qualification. Add a new scenario whenever an application introduces genuinely
new range semantics rather than merely a new name for an existing family.
