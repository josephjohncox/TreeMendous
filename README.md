# Tree-Mendous

Tree-Mendous is an **alpha** Python library for managing available integer ranges
with interchangeable Python and native CPU backends. The canonical API models
half-open ranges: `[start, end)` includes `start` and excludes `end`.

## Install

Tree-Mendous supports Python 3.11–3.13.

```bash
pip install treemendous
```

or:

```bash
uv add treemendous
```

## Quickstart

`create_range_set` requires an explicit managed domain for canonical use. The
domain starts fully available unless `initially_available=False` is supplied.

```python
from treemendous import Span, create_range_set

ranges = create_range_set(domain=(0, 24), backend="py_boundary")
ranges.discard(Span(0, 9))
ranges.discard(Span(17, 24))
booking = ranges.allocate(2, not_before=9, not_after=17)
assert booking is not None
assert (booking.start, booking.end) == (9, 11)
next_fit = ranges.first_fit(2, not_before=11, not_after=17)
assert next_fit is not None and (next_fit.start, next_fit.end) == (11, 13)
assert ranges.snapshot().total_free == 6
```

This CPU scheduling example atomically finds and removes `[9, 11)`. `allocate`
returns `None` when no fit exists. `not_after` is an exclusive scheduling
boundary: the allocated end may equal it. Empty or reversed spans and
non-positive lengths raise `ValueError`; non-integer coordinates raise
`TypeError`. Mutations outside the managed domain fail before changing state.

Payload-capable Python backends safely default to `UniformPayloadPolicy`:
adjacent or overlapping payloads merge only when equal, and `None` is ordinary
data rather than an identity. Supply an explicit policy only when non-uniform
join or ordered semantics are required:

```python
from treemendous import Span, create_range_set

cpus = create_range_set(
    domain=(0, 8),
    backend="py_boundary",
    initially_available=False,
)
cpus.add(Span(0, 8), payload="cpu")
assert cpus.first_fit(2, not_before=0).data == "cpu"
```

See [API and payload policies](docs/api.md) for join and ordered policies.
Native backends store geometry only and reject payload requirements.

## Backend maturity

Availability is probed at runtime; an explicit unavailable or invalid backend
raises a reasoned error. Automatic selection uses only stable backends that pass
semantic probes and satisfy the requested capabilities and coordinate width.

| Backend ID | Runtime | Width | Maturity | Notes |
| --- | ---: | ---: | --- | --- |
| `py_boundary` | Python/CPU | 64-bit | Stable | Core, payloads |
| `py_avl_earliest` | Python/CPU | 64-bit | Stable | Core, payloads |
| `py_summary` | Python/CPU | 64-bit | Stable | Analytics, best fit, payloads |
| `py_treap` | Python/CPU | 64-bit | Stable | Random sampling, payloads |
| `py_boundary_summary` | Python/CPU | 64-bit | Stable | Analytics, best fit, payloads |
| `cpp_boundary` | C++/CPU | 64-bit | Stable when built | Geometry only |
| `cpp_boundary_optimized` | C++/CPU | 64-bit | Stable when built | Parity alias; geometry only |
| `cpp_treap` | C++/CPU | 32-bit | Experimental | Not auto-selected |
| `cpp_boundary_summary` | C++/CPU | 32-bit | Experimental | Not auto-selected |
| `cpp_boundary_summary_optimized` | C++/CPU | 32-bit | Experimental | Not auto-selected |
| `metal_boundary_summary` | Metal/GPU | 32-bit | Experimental | macOS device gate required |
| `metal_boundary_summary_mixed` | Metal/GPU | 32-bit | Experimental | macOS device gate required |
| `gpu_boundary_summary` | CUDA/GPU | 32-bit | Experimental/unavailable | Hardware parity and sanitizer gates required |

Details and probe behavior are in [Backends](docs/backends.md).

## Maintained documentation

- [Getting started](docs/getting-started.md)
- [Canonical API and migration](docs/api.md)
- [Backend catalog](docs/backends.md)
- [Building optional backends](docs/building.md)
- [Benchmark methodology](docs/benchmarking.md)
- [Contributing and quality gates](docs/contributing.md)
- [Release process](docs/releasing.md)
- [Tracked executable example](examples/basic_rangeset.py)

## Development

```bash
uv sync --all-extras
just check
just build
```

`just check` enforces Ruff lint and formatting, mypy, branch coverage, docs
contracts, and bytecode/artifact-policy sanity. Benchmarks are separate:
`just benchmark`. See [Contributing](docs/contributing.md).

License: BSD-3-Clause.
