# Backend support

Tree-Mendous separates public interval semantics from raw geometry engines.
`RangeSet` owns validation, payload algebra, atomic mutation, and canonical
results. A backend adapter translates the small geometry protocol to one raw
implementation.

## Stable backends

Stable backends are eligible for automatic selection only after their semantic
probe passes.

| Backend ID | Runtime | Width | Native strengths |
| --- | --- | ---: | --- |
| `py_boundary` | Python CPU | 64-bit | Core geometry |
| `py_avl_earliest` | Python CPU | 64-bit | Core geometry |
| `py_summary` | Python CPU | 64-bit | Analytics and best fit |
| `py_treap` | Python CPU | 64-bit | Random sampling |
| `py_boundary_summary` | Python CPU | 64-bit | Analytics and best fit |
| `cpp_boundary` | C++ CPU | 64-bit | Core geometry |

`cpp_boundary` is present only when its compiled extension is importable and
passes the stable semantic contract. The [one-dimensional algorithm guide](theory/one_dimensional_interval_algorithms.md)
explains the boundary maps, augmented trees, operation bounds, fragmentation
costs, and native-versus-Python performance mechanisms.

## Experimental backends

| Backend ID | Runtime/device | Width | Status |
| --- | --- | ---: | --- |
| `cpp_treap` | C++ CPU | 32-bit | Cataloged, not selectable |
| `cpp_boundary_summary` | C++ CPU | 32-bit | Cataloged, not selectable |
| `cpp_boundary_summary_optimized` | C++ CPU | 32-bit | Cataloged, not selectable |
| `gpu_boundary_summary` | CUDA GPU | 32-bit | Cataloged, not selectable |
| `metal_boundary_summary` | Metal GPU | 32-bit | Cataloged, not selectable |

Experimental backends are visible for development and hardware gates but are
not accepted by the public constructor. Promotion to stable status is the only
way an implementation enters the selection surface.

## Registry and probe states

`BackendRegistry.discover()` evaluates the canonical catalog and records one
state for each backend:

- `Available`: imported and semantically validated.
- `Unavailable`: could not be imported or used on this machine.
- `Invalid`: imported but failed the semantic contract.

Selection consumes these immutable states; it does not perform discovery or
probe hardware. This makes selection deterministic and directly testable.

```python
from treemendous import BackendRegistry

registry = BackendRegistry.discover()
for spec in registry.specs:
    print(spec.id, registry.states[spec.id])
```

## Integer widths

The canonical domain and `Span` use Python integers. Each backend advertises
its native width, and construction rejects domains outside that width before
mutating native state. Stable C++ boundary operations use checked signed
64-bit arithmetic.

## Payloads

Payload policies are backend-independent. `RangeSet` maintains payload
segments while the selected backend maintains geometry, so switching between
Python and native geometry engines cannot change payload semantics.

## Promotion criteria

An experimental backend becomes stable only when all of these are true:

1. Its full operation set is implemented.
2. It passes the stable state-machine and edge-case suites.
3. It passes a real-hardware CI lane on every supported device family.
4. Its wheel resources install and load from package-relative paths.
5. It reports the same `RangeSet` results as the Python oracle.

Metal currently has a real-hardware parity lane but remains experimental.
CUDA remains quarantined until its implementation and self-hosted hardware
lane satisfy the same contract.
