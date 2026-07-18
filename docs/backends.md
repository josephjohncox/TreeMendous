# Backends

The immutable production catalog is the source of truth for IDs, maturity,
capabilities, coordinate width, and selection. Runtime probing is separate from
the catalog: a spec can be stable while unavailable in an installation.

## Stable IDs

| ID | Runtime | Capabilities |
| --- | --- | --- |
| `py_boundary` | Python CPU | core, atomic allocation, payloads |
| `py_avl_earliest` | Python CPU | core, atomic allocation, payloads |
| `py_summary` | Python CPU | core, analytics, best fit, payloads |
| `py_treap` | Python CPU | core, random sample, payloads |
| `py_boundary_summary` | Python CPU | core, analytics, best fit, payloads |
| `cpp_boundary` | C++ CPU | 64-bit core geometry |
| `cpp_boundary_optimized` | C++ CPU | 64-bit core geometry parity alias |

The optimized boundary ID is retained for compatibility; it is not a published
performance claim.

## Experimental IDs

| ID | Status |
| --- | --- |
| `cpp_treap` | 32-bit compatibility extension; capability-empty |
| `cpp_boundary_summary` | 32-bit compatibility extension; capability-empty |
| `cpp_boundary_summary_optimized` | 32-bit compatibility extension; capability-empty |
| `metal_boundary_summary` | 32-bit, requires macOS wheel/device/resource gate |
| `metal_boundary_summary_mixed` | 32-bit, requires the same Metal gate |
| `gpu_boundary_summary` | CUDA; unavailable to stable selection pending hardware parity and compute-sanitizer |

Boost ICL source can be built for dedicated validation, but no ICL backend ID is
advertised in the production catalog until that lane passes the canonical probe.

## Probe and selection behavior

A probe checks invalid-span atomicity, fragmented and containing-start first-fit,
no-fit behavior, normalized state, totals, and declared capabilities. Probe
states are `Available`, `Unavailable(reason)`, or `Invalid(error)`.

Automatic selection excludes experimental specs, failed probes, insufficient
coordinate widths, non-deterministic implementations when determinism is
requested, and missing capabilities. Explicit acquisition of an unavailable or
invalid backend raises `BackendUnavailableError` or `BackendInvalidError` with
the probe reason. `BackendRequest` and `BackendDecision` expose typed selection
requirements and rejection reasons.

Use `just test-protocols` for local diagnostics after an install or native build.
