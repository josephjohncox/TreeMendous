# Interval workload and use-case matrix

Tree-Mendous manages one-dimensional integer half-open ranges. It can represent
which coordinates are free, claimed, annotated, reserved, or overlapping. It
does not implement the surrounding search engine, regular-expression engine,
scheduler, genetic algorithm, database, network controller, or distributed
consensus protocol.

For distributed applications, Tree-Mendous can be the deterministic range-state
component inside one coordinator or a replicated state machine. The surrounding
system must still provide durable storage, fencing tokens, leases, consensus,
idempotency, worker liveness, and recovery. Sharing an in-memory `RangeSet`
between unrelated processes is not a distributed protocol.

The benchmark suite executes the following 50 application-shaped scenarios
against every stable backend. Each trace is accepted only when its complete
state, query results, mutation accounting, snapshots, statistics, and overlap
observations match the independent oracle.

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
order-preserving integer encoding. Morton or Hilbert ranges approximate spatial
queries; Tree-Mendous is not a multidimensional spatial index.

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
shared range operations work across five application families, 50 concrete
scenarios, all explicit payload policies, and every stable backend. It does not
establish:

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
