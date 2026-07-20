# Tree-Mendous documentation

## Start here

- [Getting started](getting-started.md)
- [Canonical `RangeSet` API](api.md)
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
[benchmark guide](benchmarking.md) likewise separates concrete application
evidence from generic stable-backend evidence.

## Project guides

- [Experimental multidimensional `BoxIndex` semantics](theory/box_index_denotation.md)
- [Contributing and quality gates](contributing.md)
- [Release process](releasing.md)

The stable root API is `RangeSet`, constructed with `create_range_set` or an
immutable `BackendRegistry`. Application registry APIs live under
`treemendous.applications`; concrete types live in their scenario modules. Raw
Python and native geometry modules are implementation details.
Multidimensional work remains experimental under
`treemendous.multidimensional` and is not exported from the stable root API.
