# Application engines and legacy workload matrix

Tree-Mendous ships 50 concrete, reusable application engines under
`treemendous.applications`. They implement named process-local behavior across
five packages: partitioning, scheduling, overlap catalogs, allocation, and
numeric leasing. See the [application package and API index](applications.md)
for creation, imports, shared kernels, examples, and per-scenario contracts.

Separately, the older backend benchmark suite retains 50 application-labeled
generic traces. Those traces map each scenario name to one of five `RangeSet`
operation families and replay it against every stable geometry backend. They
are backend qualification inputs, not application factories and not evidence
that a named engine works. The current inventory is therefore 50 real engines
and, separately, 50 legacy generic backend traces.

All concrete engines are in-memory and process-local. Names that include
"distributed" describe how an engine partitions or coordinates local state for
a larger system. The caller must still provide transport, durable storage,
writer handoff, downstream fencing enforcement, consensus, idempotent result
commits, worker liveness, and crash recovery. Sharing an engine or `RangeSet`
between unrelated processes is not a distributed protocol.

<!-- BEGIN GENERATED SCENARIO STATUS -->
## Application implementation status

Current completion: **50/50** real engines.
A legacy backend trace is not implementation evidence. An entry becomes
`COMPLETE` only when its engine, example, independent oracle, benchmark,
and scenario documentation are all registered and resolve.

| Scenario | Family | Category | Status | Engine | Example | Oracle | Benchmark | Docs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `distributed-document-search` | `partition` | `distributed_partition` | `COMPLETE` | [create_document_search](../treemendous/applications/partitioning/document_search.py) | [example](../examples/applications/partitioning/document_search.py) | [oracle](../tests/oracles/applications/partitioning/document_search.py) | [benchmark](../tests/performance/applications/partitioning/document_search.py) | [docs](scenarios/partitioning/document_search.md) |
| `distributed-regex-scan` | `partition` | `distributed_partition` | `COMPLETE` | [create_regex_scan](../treemendous/applications/partitioning/regex_scan.py) | [example](../examples/applications/partitioning/regex_scan.py) | [oracle](../tests/oracles/applications/partitioning/regex_scan.py) | [benchmark](../tests/performance/applications/partitioning/regex_scan.py) | [docs](scenarios/partitioning/regex_scan.md) |
| `distributed-genetic-search` | `partition` | `distributed_partition` | `COMPLETE` | [create_genetic_search](../treemendous/applications/partitioning/genetic_search.py) | [example](../examples/applications/partitioning/genetic_search.py) | [oracle](../tests/oracles/applications/partitioning/genetic_search.py) | [benchmark](../tests/performance/applications/partitioning/genetic_search.py) | [docs](scenarios/partitioning/genetic_search.md) |
| `distributed-graph-search` | `partition` | `distributed_partition` | `COMPLETE` | [create_graph_search](../treemendous/applications/partitioning/graph_search.py) | [example](../examples/applications/partitioning/graph_search.py) | [oracle](../tests/oracles/applications/partitioning/graph_search.py) | [benchmark](../tests/performance/applications/partitioning/graph_search.py) | [docs](scenarios/partitioning/graph_search.md) |
| `distributed-sat-search` | `partition` | `distributed_partition` | `COMPLETE` | [create_sat_search](../treemendous/applications/partitioning/sat_search.py) | [example](../examples/applications/partitioning/sat_search.py) | [oracle](../tests/oracles/applications/partitioning/sat_search.py) | [benchmark](../tests/performance/applications/partitioning/sat_search.py) | [docs](scenarios/partitioning/sat_search.md) |
| `distributed-fuzzing` | `partition` | `distributed_partition` | `COMPLETE` | [create_fuzzing](../treemendous/applications/partitioning/fuzzing.py) | [example](../examples/applications/partitioning/fuzzing.py) | [oracle](../tests/oracles/applications/partitioning/fuzzing.py) | [benchmark](../tests/performance/applications/partitioning/fuzzing.py) | [docs](scenarios/partitioning/fuzzing.md) |
| `distributed-hyperparameter-search` | `partition` | `distributed_partition` | `COMPLETE` | [create_hyperparameter_search](../treemendous/applications/partitioning/hyperparameter_search.py) | [example](../examples/applications/partitioning/hyperparameter_search.py) | [oracle](../tests/oracles/applications/partitioning/hyperparameter_search.py) | [benchmark](../tests/performance/applications/partitioning/hyperparameter_search.py) | [docs](scenarios/partitioning/hyperparameter_search.md) |
| `distributed-log-replay` | `partition` | `distributed_partition` | `COMPLETE` | [create_log_replay](../treemendous/applications/partitioning/log_replay.py) | [example](../examples/applications/partitioning/log_replay.py) | [oracle](../tests/oracles/applications/partitioning/log_replay.py) | [benchmark](../tests/performance/applications/partitioning/log_replay.py) | [docs](scenarios/partitioning/log_replay.md) |
| `distributed-build-sharding` | `partition` | `distributed_partition` | `COMPLETE` | [create_build_sharding](../treemendous/applications/partitioning/build_sharding.py) | [example](../examples/applications/partitioning/build_sharding.py) | [oracle](../tests/oracles/applications/partitioning/build_sharding.py) | [benchmark](../tests/performance/applications/partitioning/build_sharding.py) | [docs](scenarios/partitioning/build_sharding.md) |
| `map-reduce-input-splits` | `partition` | `distributed_partition` | `COMPLETE` | [create_map_reduce](../treemendous/applications/partitioning/map_reduce.py) | [example](../examples/applications/partitioning/map_reduce.py) | [oracle](../tests/oracles/applications/partitioning/map_reduce.py) | [benchmark](../tests/performance/applications/partitioning/map_reduce.py) | [docs](scenarios/partitioning/map_reduce.md) |
| `distributed-web-crawl` | `partition` | `distributed_partition` | `COMPLETE` | [create_web_crawl](../treemendous/applications/partitioning/web_crawl.py) | [example](../examples/applications/partitioning/web_crawl.py) | [oracle](../tests/oracles/applications/partitioning/web_crawl.py) | [benchmark](../tests/performance/applications/partitioning/web_crawl.py) | [docs](scenarios/partitioning/web_crawl.md) |
| `distributed-index-merge` | `partition` | `distributed_partition` | `COMPLETE` | [create_index_merge](../treemendous/applications/partitioning/index_merge.py) | [example](../examples/applications/partitioning/index_merge.py) | [oracle](../tests/oracles/applications/partitioning/index_merge.py) | [benchmark](../tests/performance/applications/partitioning/index_merge.py) | [docs](scenarios/partitioning/index_merge.md) |
| `distributed-cluster-scheduling` | `scheduling` | `scheduling_reservation` | `COMPLETE` | [create_cluster_scheduler](../treemendous/applications/scheduling/cluster.py) | [example](../examples/applications/scheduling/cluster.py) | [oracle](../tests/oracles/applications/scheduling/cluster.py) | [benchmark](../tests/performance/applications/scheduling/test_cluster_smoke.py) | [docs](scenarios/scheduling/distributed-cluster-scheduling.md) |
| `gpu-stream-scheduling` | `scheduling` | `scheduling_reservation` | `COMPLETE` | [create_gpu_stream_scheduler](../treemendous/applications/scheduling/gpu_streams.py) | [example](../examples/applications/scheduling/gpu_streams.py) | [oracle](../tests/oracles/applications/scheduling/gpu_streams.py) | [benchmark](../tests/performance/applications/scheduling/test_gpu_streams_smoke.py) | [docs](scenarios/scheduling/gpu-stream-scheduling.md) |
| `render-farm-frames` | `scheduling` | `scheduling_reservation` | `COMPLETE` | [create_render_farm_scheduler](../treemendous/applications/scheduling/render_farm.py) | [example](../examples/applications/scheduling/render_farm.py) | [oracle](../tests/oracles/applications/scheduling/render_farm.py) | [benchmark](../tests/performance/applications/scheduling/test_render_farm_smoke.py) | [docs](scenarios/scheduling/render-farm-frames.md) |
| `ci-runner-reservations` | `scheduling` | `scheduling_reservation` | `COMPLETE` | [create_ci_runner_scheduler](../treemendous/applications/scheduling/ci_runners.py) | [example](../examples/applications/scheduling/ci_runners.py) | [oracle](../tests/oracles/applications/scheduling/ci_runners.py) | [benchmark](../tests/performance/applications/scheduling/test_ci_runners_smoke.py) | [docs](scenarios/scheduling/ci-runner-reservations.md) |
| `meeting-room-booking` | `scheduling` | `scheduling_reservation` | `COMPLETE` | [create_meeting_room_scheduler](../treemendous/applications/scheduling/meeting_rooms.py) | [example](../examples/applications/scheduling/meeting_rooms.py) | [oracle](../tests/oracles/applications/scheduling/meeting_rooms.py) | [benchmark](../tests/performance/applications/scheduling/test_meeting_rooms_smoke.py) | [docs](scenarios/scheduling/meeting-room-booking.md) |
| `airline-gate-assignment` | `scheduling` | `scheduling_reservation` | `COMPLETE` | [create_airline_gate_scheduler](../treemendous/applications/scheduling/airline_gates.py) | [example](../examples/applications/scheduling/airline_gates.py) | [oracle](../tests/oracles/applications/scheduling/airline_gates.py) | [benchmark](../tests/performance/applications/scheduling/test_airline_gates_smoke.py) | [docs](scenarios/scheduling/airline-gate-assignment.md) |
| `operating-room-booking` | `scheduling` | `scheduling_reservation` | `COMPLETE` | [create_operating_room_scheduler](../treemendous/applications/scheduling/operating_rooms.py) | [example](../examples/applications/scheduling/operating_rooms.py) | [oracle](../tests/oracles/applications/scheduling/operating_rooms.py) | [benchmark](../tests/performance/applications/scheduling/test_operating_rooms_smoke.py) | [docs](scenarios/scheduling/operating-room-booking.md) |
| `laboratory-equipment-booking` | `scheduling` | `scheduling_reservation` | `COMPLETE` | [create_lab_instrument_scheduler](../treemendous/applications/scheduling/lab_instruments.py) | [example](../examples/applications/scheduling/lab_instruments.py) | [oracle](../tests/oracles/applications/scheduling/lab_instruments.py) | [benchmark](../tests/performance/applications/scheduling/test_lab_instruments_smoke.py) | [docs](scenarios/scheduling/laboratory-equipment-booking.md) |
| `fleet-charging-windows` | `scheduling` | `scheduling_reservation` | `COMPLETE` | [create_fleet_charging_scheduler](../treemendous/applications/scheduling/fleet_charging.py) | [example](../examples/applications/scheduling/fleet_charging.py) | [oracle](../tests/oracles/applications/scheduling/fleet_charging.py) | [benchmark](../tests/performance/applications/scheduling/test_fleet_charging_smoke.py) | [docs](scenarios/scheduling/fleet-charging-windows.md) |
| `radio-spectrum-timeslots` | `scheduling` | `scheduling_reservation` | `COMPLETE` | [create_radio_spectrum_scheduler](../treemendous/applications/scheduling/radio_spectrum.py) | [example](../examples/applications/scheduling/radio_spectrum.py) | [oracle](../tests/oracles/applications/scheduling/radio_spectrum.py) | [benchmark](../tests/performance/applications/scheduling/test_radio_spectrum_smoke.py) | [docs](scenarios/scheduling/radio-spectrum-timeslots.md) |
| `warehouse-dock-appointments` | `scheduling` | `scheduling_reservation` | `COMPLETE` | [create_warehouse_dock_scheduler](../treemendous/applications/scheduling/warehouse_docks.py) | [example](../examples/applications/scheduling/warehouse_docks.py) | [oracle](../tests/oracles/applications/scheduling/warehouse_docks.py) | [benchmark](../tests/performance/applications/scheduling/test_warehouse_docks_smoke.py) | [docs](scenarios/scheduling/warehouse-dock-appointments.md) |
| `maintenance-window-planning` | `scheduling` | `scheduling_reservation` | `COMPLETE` | [create_maintenance_scheduler](../treemendous/applications/scheduling/maintenance.py) | [example](../examples/applications/scheduling/maintenance.py) | [oracle](../tests/oracles/applications/scheduling/maintenance.py) | [benchmark](../tests/performance/applications/scheduling/test_maintenance_smoke.py) | [docs](scenarios/scheduling/maintenance-window-planning.md) |
| `genomic-annotation-overlap` | `catalog` | `overlap_catalog` | `COMPLETE` | [create_catalog](../treemendous/applications/catalogs/genomic_annotation_overlap.py) | [example](../examples/applications/catalogs/genomic_annotation_overlap.py) | [oracle](../tests/oracles/applications/catalogs/genomic_annotation_overlap.py) | [benchmark](../tests/performance/applications/catalogs/genomic_annotation_overlap.py) | [docs](scenarios/catalogs/genomic-annotation-overlap.md) |
| `source-diagnostic-ranges` | `catalog` | `overlap_catalog` | `COMPLETE` | [create_catalog](../treemendous/applications/catalogs/source_diagnostic_ranges.py) | [example](../examples/applications/catalogs/source_diagnostic_ranges.py) | [oracle](../tests/oracles/applications/catalogs/source_diagnostic_ranges.py) | [benchmark](../tests/performance/applications/catalogs/source_diagnostic_ranges.py) | [docs](scenarios/catalogs/source-diagnostic-ranges.md) |
| `filesystem-byte-locks` | `catalog` | `overlap_catalog` | `COMPLETE` | [create_lock_table](../treemendous/applications/catalogs/filesystem_byte_locks.py) | [example](../examples/applications/catalogs/filesystem_byte_locks.py) | [oracle](../tests/oracles/applications/catalogs/filesystem_byte_locks.py) | [benchmark](../tests/performance/applications/catalogs/filesystem_byte_locks.py) | [docs](scenarios/catalogs/filesystem-byte-locks.md) |
| `database-key-range-locks` | `catalog` | `overlap_catalog` | `COMPLETE` | [create_lock_table](../treemendous/applications/catalogs/database_key_range_locks.py) | [example](../examples/applications/catalogs/database_key_range_locks.py) | [oracle](../tests/oracles/applications/catalogs/database_key_range_locks.py) | [benchmark](../tests/performance/applications/catalogs/database_key_range_locks.py) | [docs](scenarios/catalogs/database-key-range-locks.md) |
| `packet-sequence-reassembly` | `catalog` | `overlap_catalog` | `COMPLETE` | [create_catalog](../treemendous/applications/catalogs/packet_sequence_reassembly.py) | [example](../examples/applications/catalogs/packet_sequence_reassembly.py) | [oracle](../tests/oracles/applications/catalogs/packet_sequence_reassembly.py) | [benchmark](../tests/performance/applications/catalogs/packet_sequence_reassembly.py) | [docs](scenarios/catalogs/packet-sequence-reassembly.md) |
| `subtitle-cue-ranges` | `catalog` | `overlap_catalog` | `COMPLETE` | [create_catalog](../treemendous/applications/catalogs/subtitle_cue_ranges.py) | [example](../examples/applications/catalogs/subtitle_cue_ranges.py) | [oracle](../tests/oracles/applications/catalogs/subtitle_cue_ranges.py) | [benchmark](../tests/performance/applications/catalogs/subtitle_cue_ranges.py) | [docs](scenarios/catalogs/subtitle-cue-ranges.md) |
| `video-edit-regions` | `catalog` | `overlap_catalog` | `COMPLETE` | [create_catalog](../treemendous/applications/catalogs/video_edit_regions.py) | [example](../examples/applications/catalogs/video_edit_regions.py) | [oracle](../tests/oracles/applications/catalogs/video_edit_regions.py) | [benchmark](../tests/performance/applications/catalogs/video_edit_regions.py) | [docs](scenarios/catalogs/video-edit-regions.md) |
| `timeseries-alert-windows` | `catalog` | `overlap_catalog` | `COMPLETE` | [create_catalog](../treemendous/applications/catalogs/timeseries_alert_windows.py) | [example](../examples/applications/catalogs/timeseries_alert_windows.py) | [oracle](../tests/oracles/applications/catalogs/timeseries_alert_windows.py) | [benchmark](../tests/performance/applications/catalogs/timeseries_alert_windows.py) | [docs](scenarios/catalogs/timeseries-alert-windows.md) |
| `distributed-trace-spans` | `catalog` | `overlap_catalog` | `COMPLETE` | [create_catalog](../treemendous/applications/catalogs/distributed_trace_spans.py) | [example](../examples/applications/catalogs/distributed_trace_spans.py) | [oracle](../tests/oracles/applications/catalogs/distributed_trace_spans.py) | [benchmark](../tests/performance/applications/catalogs/distributed_trace_spans.py) | [docs](scenarios/catalogs/distributed-trace-spans.md) |
| `morton-geospatial-ranges` | `catalog` | `overlap_catalog` | `COMPLETE` | [create_catalog](../treemendous/applications/catalogs/morton_geospatial_ranges.py) | [example](../examples/applications/catalogs/morton_geospatial_ranges.py) | [oracle](../tests/oracles/applications/catalogs/morton_geospatial_ranges.py) | [benchmark](../tests/performance/applications/catalogs/morton_geospatial_ranges.py) | [docs](scenarios/catalogs/morton-geospatial-ranges.md) |
| `heap-free-space` | `allocator` | `allocation_churn` | `COMPLETE` | [create_application](../treemendous/applications/allocation/heap.py) | [example](../examples/applications/allocation/heap_free_space.py) | [oracle](../tests/oracles/applications/allocation/heap_free_space.py) | [benchmark](../tests/performance/applications/allocation/heap_free_space.py) | [docs](scenarios/allocation/heap_free_space.md) |
| `disk-block-allocation` | `allocator` | `allocation_churn` | `COMPLETE` | [create_application](../treemendous/applications/allocation/disk_blocks.py) | [example](../examples/applications/allocation/disk_block_allocation.py) | [oracle](../tests/oracles/applications/allocation/disk_block_allocation.py) | [benchmark](../tests/performance/applications/allocation/disk_block_allocation.py) | [docs](scenarios/allocation/disk_block_allocation.md) |
| `virtual-address-space` | `allocator` | `allocation_churn` | `COMPLETE` | [create_application](../treemendous/applications/allocation/virtual_address.py) | [example](../examples/applications/allocation/virtual_address_space.py) | [oracle](../tests/oracles/applications/allocation/virtual_address_space.py) | [benchmark](../tests/performance/applications/allocation/virtual_address_space.py) | [docs](scenarios/allocation/virtual_address_space.md) |
| `database-page-allocation` | `allocator` | `allocation_churn` | `COMPLETE` | [create_application](../treemendous/applications/allocation/database_pages.py) | [example](../examples/applications/allocation/database_page_allocation.py) | [oracle](../tests/oracles/applications/allocation/database_page_allocation.py) | [benchmark](../tests/performance/applications/allocation/database_page_allocation.py) | [docs](scenarios/allocation/database_page_allocation.md) |
| `object-store-multipart-ranges` | `allocator` | `allocation_churn` | `COMPLETE` | [create_application](../treemendous/applications/allocation/multipart_upload.py) | [example](../examples/applications/allocation/object_store_multipart_ranges.py) | [oracle](../tests/oracles/applications/allocation/object_store_multipart_ranges.py) | [benchmark](../tests/performance/applications/allocation/object_store_multipart_ranges.py) | [docs](scenarios/allocation/object_store_multipart_ranges.md) |
| `cdn-byte-range-cache` | `allocator` | `allocation_churn` | `COMPLETE` | [create_application](../treemendous/applications/allocation/cdn_cache.py) | [example](../examples/applications/allocation/cdn_byte_range_cache.py) | [oracle](../tests/oracles/applications/allocation/cdn_byte_range_cache.py) | [benchmark](../tests/performance/applications/allocation/cdn_byte_range_cache.py) | [docs](scenarios/allocation/cdn_byte_range_cache.md) |
| `gpu-memory-arena` | `allocator` | `allocation_churn` | `COMPLETE` | [create_application](../treemendous/applications/allocation/gpu_arena.py) | [example](../examples/applications/allocation/gpu_memory_arena.py) | [oracle](../tests/oracles/applications/allocation/gpu_memory_arena.py) | [benchmark](../tests/performance/applications/allocation/gpu_memory_arena.py) | [docs](scenarios/allocation/gpu_memory_arena.md) |
| `ring-buffer-sequences` | `allocator` | `allocation_churn` | `COMPLETE` | [create_application](../treemendous/applications/allocation/ring_buffer.py) | [example](../examples/applications/allocation/ring_buffer_sequences.py) | [oracle](../tests/oracles/applications/allocation/ring_buffer_sequences.py) | [benchmark](../tests/performance/applications/allocation/ring_buffer_sequences.py) | [docs](scenarios/allocation/ring_buffer_sequences.md) |
| `tcp-udp-port-leases` | `lease` | `resource_leasing` | `COMPLETE` | [create_engine](../treemendous/applications/leasing/tcp_udp_ports.py) | [example](../examples/applications/leasing/tcp_udp_port_leases.py) | [oracle](../tests/oracles/applications/leasing/tcp_udp_port_leases.py) | [benchmark](../tests/performance/applications/leasing/tcp_udp_port_leases.py) | [docs](scenarios/leasing/tcp_udp_port_leases.md) |
| `numeric-ip-address-pools` | `lease` | `resource_leasing` | `COMPLETE` | [create_engine](../treemendous/applications/leasing/numeric_ip_pools.py) | [example](../examples/applications/leasing/numeric_ip_address_pools.py) | [oracle](../tests/oracles/applications/leasing/numeric_ip_address_pools.py) | [benchmark](../tests/performance/applications/leasing/numeric_ip_address_pools.py) | [docs](scenarios/leasing/numeric_ip_address_pools.md) |
| `database-id-pools` | `lease` | `resource_leasing` | `COMPLETE` | [create_engine](../treemendous/applications/leasing/database_ids.py) | [example](../examples/applications/leasing/database_id_pools.py) | [oracle](../tests/oracles/applications/leasing/database_id_pools.py) | [benchmark](../tests/performance/applications/leasing/database_id_pools.py) | [docs](scenarios/leasing/database_id_pools.md) |
| `software-license-seats` | `lease` | `resource_leasing` | `COMPLETE` | [create_engine](../treemendous/applications/leasing/software_seats.py) | [example](../examples/applications/leasing/software_license_seats.py) | [oracle](../tests/oracles/applications/leasing/software_license_seats.py) | [benchmark](../tests/performance/applications/leasing/software_license_seats.py) | [docs](scenarios/leasing/software_license_seats.md) |
| `warehouse-bin-ranges` | `lease` | `resource_leasing` | `COMPLETE` | [create_engine](../treemendous/applications/leasing/warehouse_bins.py) | [example](../examples/applications/leasing/warehouse_bin_ranges.py) | [oracle](../tests/oracles/applications/leasing/warehouse_bin_ranges.py) | [benchmark](../tests/performance/applications/leasing/warehouse_bin_ranges.py) | [docs](scenarios/leasing/warehouse_bin_ranges.md) |
| `game-world-region-ids` | `lease` | `resource_leasing` | `COMPLETE` | [create_engine](../treemendous/applications/leasing/game_regions.py) | [example](../examples/applications/leasing/game_world_region_ids.py) | [oracle](../tests/oracles/applications/leasing/game_world_region_ids.py) | [benchmark](../tests/performance/applications/leasing/game_world_region_ids.py) | [docs](scenarios/leasing/game_world_region_ids.md) |
| `vlan-tag-pools` | `lease` | `resource_leasing` | `COMPLETE` | [create_engine](../treemendous/applications/leasing/vlan_tags.py) | [example](../examples/applications/leasing/vlan_tag_pools.py) | [oracle](../tests/oracles/applications/leasing/vlan_tag_pools.py) | [benchmark](../tests/performance/applications/leasing/vlan_tag_pools.py) | [docs](scenarios/leasing/vlan_tag_pools.md) |
| `phone-extension-pools` | `lease` | `resource_leasing` | `COMPLETE` | [create_engine](../treemendous/applications/leasing/phone_extensions.py) | [example](../examples/applications/leasing/phone_extension_pools.py) | [oracle](../tests/oracles/applications/leasing/phone_extension_pools.py) | [benchmark](../tests/performance/applications/leasing/phone_extension_pools.py) | [docs](scenarios/leasing/phone_extension_pools.md) |
<!-- END GENERATED SCENARIO STATUS -->

## Legacy generic backend qualification traces

The benchmark suite executes the following 50 application-shaped scenarios
against every stable backend. These are deliberately generic range-operation
qualifications mapped onto application names. They do not call the concrete
engines and cannot mark a manifest entry complete. Each trace is accepted only
when its complete state, query results, mutation accounting, snapshots,
statistics, and overlap observations match the independent range oracle.

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

## What the legacy matrix does not prove

No finite suite proves every possible use case. This legacy matrix establishes
that shared range operations work across five operation families, 50 labeled
generic traces, all explicit payload policies, and every stable backend. It does
not execute or validate a concrete application engine. For that evidence, use
the separate
[`application_benchmark_suite`](benchmarking.md#concrete-application-suite).
The legacy matrix also does not establish:

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
engineering evidence, and the sharded large profile for bounded
high-cardinality observations. Add a generic trace only when a workload
introduces genuinely new range semantics rather than merely a new name for an
existing family.
