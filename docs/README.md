# Tree-Mendous documentation

- [Getting started](getting-started.md)
- [Canonical API](api.md)
- [Backend support](backends.md)
- [Building optional backends](building.md)
- [Benchmark methodology](benchmarking.md)
- [Application-shaped workloads and implementation status](use-cases.md)
- [Experimental multidimensional `BoxIndex` semantics](theory/box_index_denotation.md)
- [Contributing and quality gates](contributing.md)
- [Release process](releasing.md)

The public API is `RangeSet`, constructed with `create_range_set` or an
immutable `BackendRegistry`. Raw Python and native geometry modules are
implementation details. Multidimensional work remains experimental under
`treemendous.multidimensional` and is not exported from the stable root API.
