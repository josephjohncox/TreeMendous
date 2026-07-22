# Tree-Mendous documentation

## Start here

- [Getting started](getting-started.md)
- [Choosing an interface](choosing-an-interface.md)
- [Canonical `RangeSet` API](api.md)
- [Performance](performance.md)
- [Backend support](backends.md)
- [Building optional backends](building.md)

## Application engines

The [application index](applications.md) documents the process-local API,
shared kernels, guarantees, exclusions, and all 50 concrete engines. Browse by
family:

- [Partitioning and work claiming](applications.md#partitioning-and-work-claiming)
- [Scheduling and reservation](applications.md#scheduling-and-reservation)
- [Identity-preserving overlap catalogs](applications.md#identity-preserving-overlap-catalogs)
- [Allocation and capacity tracking](applications.md#allocation-and-capacity-tracking)
- [Numeric resource leasing](applications.md#numeric-resource-leasing)

The [implementation status and legacy workload matrix](use-cases.md) separates
real engines from the 50 generic backend qualification traces. The
[application-pattern guide](application-patterns.md) shows where the separate
exact-batch and experimental multidimensional APIs fit, with explicit
exclusions. The [benchmark guide](benchmarking.md) separates concrete
application evidence from generic stable-backend evidence, while the
[user-facing performance guide](performance.md) reports scoped measurements and
the optimization roadmap.

## Project guides

- [One-dimensional interval formal model](theory/one_dimensional_interval_formal_model.md)
- [One-dimensional interval algorithms and performance](theory/one_dimensional_interval_algorithms.md)
- [Experimental multidimensional `BoxIndex` semantics](theory/box_index_denotation.md)
- [Experimental optimized multidimensional indexes](theory/optimized_box_indexes.md)
- [Stable specialized exact whole-batch CPU mutations](exact-batch.md)
- [Multidimensional batch, SIMD, Metal, and CUDA design](theory/multidimensional_acceleration.md)
- [Contributing and quality gates](contributing.md)
- [Release process](releasing.md)
- [Tree-Mendous 1.1.1 release notes](releases/1.1.1.md)
- [Tree-Mendous 1.1.0 release notes](releases/1.1.0.md)

The stable root API is `RangeSet`, constructed with `create_range_set` or an
immutable `BackendRegistry`. Application registry APIs live under
`treemendous.applications`; concrete types live in their scenario modules. Raw
Python and native geometry modules are implementation details.
Multidimensional work remains experimental under
`treemendous.multidimensional` and is not exported from the stable root API.
