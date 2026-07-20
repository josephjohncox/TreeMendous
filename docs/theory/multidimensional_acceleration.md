# Multidimensional batch acceleration

The experimental multidimensional indexes currently execute on the CPU. GPU,
Metal, and vectorized CPU work is useful only for workload shapes that amortize
packing, dispatch, synchronization, and result reconstruction. This document
defines the acceleration boundary and the correctness constraints for future
implementations. It is not a claim that a device backend exists today.

## The operation to accelerate

For `n` stored boxes in fixed dimension `d` and `m` query boxes, exact overlap
is the Boolean matrix

```text
M[q, i] = all(
    entry_lower[i, axis] < query_upper[q, axis]
    and query_lower[q, axis] < entry_upper[i, axis]
    for axis in 0..d-1
).
```

The calculation is regular when lower and upper coordinates are stored as
contiguous structure-of-arrays buffers:

```text
entry_lower[d][n]
entry_upper[d][n]
query_lower[d][m]
query_upper[d][m]
```

It is irregular when executed against Python `Box` objects, dictionaries,
sparse-grid cell tuples, or projection trees. The first acceleration target
should therefore be batch exact filtering over a captured immutable snapshot,
not device-side mutation of the existing Python indexes.

## Semantics that cannot change

An accelerated result must refine `BoxIndex.overlaps` exactly:

- integer half-open comparisons on every axis;
- no coordinate truncation or floating-point conversion;
- equal boxes remain distinct entries;
- owner-scoped handles retain their identities;
- results are ordered by query, then by insertion sequence;
- payloads are cloned only after exact geometry succeeds;
- one result batch observes one committed index version;
- validation, allocation, upload, execution, download, and payload-clone
  failures publish no partial result or mutation.

Atomic append order from GPU threads is not insertion order. A kernel that uses
an atomic counter to append matches produces schedule-dependent output and does
not satisfy the contract.

## Coordinate widths

The Python model accepts exact, unbounded integers. Native vector lanes and
device kernels use fixed-width values. Preparation must prove that every stored
and query coordinate fits the selected signed type before casting.

A practical initial policy is:

| Engine | Candidate coordinate type | Required guard |
| --- | --- | --- |
| C++ scalar/SIMD | signed 64-bit | exact range preflight |
| CUDA | signed 64-bit | exact range preflight and checked packing |
| Metal | signed 32- or 64-bit, device-dependent | explicit capability and range preflight |

Unsupported coordinates must select the exact CPU path or raise according to an
explicit option. Silent narrowing is invalid.

## Stable result construction

A batch kernel should produce one of two deterministic intermediate forms:

1. a dense bit mask in entry-slot order; or
2. stable-compacted entry slots produced by a deterministic prefix sum.

The host then maps slots to handles in the captured snapshot. Returning sequence
numbers instead of device-generated handles keeps owner identity and payload
objects out of device memory.

For a mask, host compaction naturally preserves insertion order. Its transfer
cost is `ceil(n / 8)` bytes per query when bit-packed, before protocol overhead.
For dense results, transferring all matching slots can cost more than the mask.
The representation should be selected from measured result density, not from a
fixed universal rule.

## Snapshot-oriented architecture

The useful internal object is a prepared, versioned query snapshot:

```text
PreparedBoxes
  owner identity
  captured version
  dimensions
  coordinate type
  lower/upper structure-of-arrays buffers
  slot -> BoxHandle mapping
  selected engine
  retained host/device bytes
```

Preparation captures one immutable `_PublishedState`. A later index mutation
does not mutate or silently refresh the prepared object. The old prepared
snapshot remains valid for its captured version until explicitly released.

A future experimental API could take this shape:

```python
prepared = index.prepare_batch_queries(
    backend="auto",              # scalar, cpu_vectorized, metal, cuda
    on_unsupported="cpu",        # cpu or raise
)

matches = prepared.overlaps_many(
    queries,
    return_type="handles",       # handles, entries, or mask
)
```

`entries` would perform the normal payload clones on the host after geometry
completion. The API should remain outside the stable root namespace until it
has independent parity, failure, packaging, and hardware evidence.

## CPU vectorization first

Fixed 2D, 3D, and 4D comparisons over contiguous `int64_t` arrays are good
compiler-vectorization candidates. A native C++ batch loop has several
advantages over a first GPU implementation:

- no device transfer or synchronization;
- the same signed 64-bit representation as `cpp_boundary`;
- ordinary sanitizer and CPU CI coverage;
- low cold-start cost;
- easier deterministic slot-order output;
- useful speedups at smaller `n * m` products.

The first implementation should use tiled scalar C++ with structure-of-arrays
buffers and inspect compiler vectorization reports before adding architecture
intrinsics. Explicit AVX or NEON code adds portability and dispatch work and is
not justified until auto-vectorization fails a measured bottleneck.

Tiling bounds temporary masks. For tile sizes `N_t` and `M_t`, temporary Boolean
storage is proportional to `N_t * M_t`, not the full `n * m` matrix.

## Metal

Apple unified memory reduces explicit copy cost, so Metal is a plausible first
device experiment after the C++ batch reference. It does not eliminate command
encoding, dispatch, synchronization, or cache-coherency costs.

A Metal path should:

1. keep prepared entry buffers resident across many query batches;
2. validate the supported integer width before dispatch;
3. write masks or stable prefix-sum-compacted slots;
4. avoid atomic append order;
5. cap buffer products and output bytes before allocation;
6. compare cold and warm execution separately;
7. run exact parity on real Apple hardware.

The existing Metal boundary-summary kernels demonstrate packaging and hardware
wiring, but their 32-bit scalar assumptions and atomic output patterns are not a
multidimensional semantic foundation.

## CUDA

CUDA becomes attractive for large resident snapshots and repeated batches. A
useful implementation requires:

- persistent device entry buffers;
- pinned or otherwise measured transfer buffers;
- asynchronous upload, kernel, and download where ordering permits;
- signed 64-bit comparison parity;
- a deterministic compaction path;
- compute-sanitizer coverage;
- an explicit minimum-work threshold.

A scalar query against a small or frequently changing index is a bad CUDA
workload. Kernel launch and synchronization can exceed the complete CPU query.

## Projection and sparse-grid interaction

The current fixed-dimensional projection and bounded sparse-grid indexes are
candidate generators. Their internal structures are pointer-heavy or hash-based
and update through immutable Python state. Porting those mutation structures to
a GPU would require a separate data-structure design.

A hybrid path is more credible:

1. the CPU projection or grid selects candidate slots;
2. the batch engine performs exact all-axis comparisons on those slots;
3. the host restores insertion order and clones payloads.

This helps only when candidate batches are large and dense enough. If a sparse
grid returns five candidates, scalar CPU rechecks are cheaper than packing and
dispatch. If it returns most of a million-entry snapshot for thousands of
queries, vectorized filtering can pay.

## Cost and crossover model

A device batch wins only when

```text
T_prepare_amortized
+ T_query_pack
+ T_dispatch
+ T_compare
+ T_result_transfer
+ T_host_reconstruction
< T_exact_cpu_query.
```

The crossover depends on:

- `n`, `m`, and `d`;
- mutation-to-query ratio;
- snapshot reuse count;
- candidate selectivity;
- result density;
- coordinate width;
- payload clone cost;
- cold versus resident buffers;
- device and driver.

`backend="auto"` must use measured, conservative thresholds tied to these
variables. It must report why it selected or rejected an accelerator. It must
not hide device failure behind an unreported fallback.

## Resource guardrails

Preparation and each batch need explicit limits for:

- entry count;
- query count;
- dimension;
- `n * m` comparison product;
- retained host bytes;
- retained device bytes;
- temporary mask bytes;
- maximum returned slots;
- coordinate-width compatibility.

All products must be checked before allocation or dispatch. A rejected batch
must leave the prepared snapshot and source index unchanged.

## Evidence required before promotion

An accelerator needs more than a fast kernel microbenchmark:

1. differential results against linear `BoxIndex` for duplicates, updates,
   removals, ordering, and boundary contact;
2. state-machine tests over prepared and stale snapshots;
3. injected allocation, upload, kernel, synchronization, and clone failures;
4. arbitrary-coordinate rejection and fixed-width boundary tests;
5. cold/warm and scalar/batch crossover measurements;
6. result-density and selectivity sweeps;
7. memory-limit tests before every product allocation;
8. real Metal/CUDA hardware parity and sanitizer lanes;
9. clean-wheel resource and arbitrary-working-directory installation tests.

Until those gates pass, CPU indexes remain authoritative and device work remains
an explicit experiment.
