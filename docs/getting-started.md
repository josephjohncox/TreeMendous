# Getting started

## Requirements

- Python 3.11 or newer
- `uv` for the repository workflow
- A C++17 compiler for native CPU extensions
- Optional: Xcode command-line tools for Metal builds
- Optional: a CUDA toolkit and NVIDIA GPU for CUDA experiments

## Install from PyPI

```bash
python -m pip install treemendous
```

## Install for development

```bash
git clone https://github.com/josephjohncox/TreeMendous.git
cd TreeMendous
uv sync --all-extras
uv run pytest -q
```

## Create a range set

```python
from treemendous import Span, create_range_set

ranges = create_range_set((0, 1_000), initially_available=True)
ranges.discard(Span(100, 200))
slot = ranges.allocate(50, not_before=150)

assert slot is not None and slot.span == Span(200, 250)
```

Intervals are half-open. Reserving `[100, 200)` removes exactly the integers
from 100 through 199.

## Choose a backend

Omitting `backend` selects only a stable backend whose semantic probe passed.
Select a stable backend explicitly when reproducibility matters:

```python
ranges = create_range_set((0, 1_000), backend="py_boundary")
```

Inspect discovery results with an immutable registry:

```python
from treemendous import BackendRegistry

registry = BackendRegistry.discover()
for spec in registry.available_specs():
    print(spec.id, spec.runtime, spec.device)
```

Experimental implementations are cataloged for development and hardware gates
but are not accepted by the public constructor.

## Attach payloads

Payload behavior requires an explicit policy:

```python
from treemendous import Span, UniformPayloadPolicy, create_range_set

ranges = create_range_set(
    (0, 100),
    initially_available=False,
    payload_policy=UniformPayloadPolicy(),
)
ranges.add(Span(10, 20), payload="maintenance")

assert tuple((result.span, result.data) for result in ranges.intervals()) == (
    (Span(10, 20), "maintenance"),
)
```

See [API](api.md) for join and ordered policies.

## Run repository checks

```bash
just test
just check
just validate
```

For directional benchmark data:

```bash
just test-perf
```

Benchmark results are correctness-checked before they are reported. They are
local measurements, not universal performance claims.

## Build artifacts

```bash
just build
uv run python scripts/verify_artifact_contents.py dist/*.whl dist/*.tar.gz
```

On macOS, build and test the experimental Metal extension with:

```bash
just build-metal
just test-metal
```

On a CUDA host:

```bash
just build-gpu
just test-gpu
```

## Next steps

- [Canonical API](api.md)
- [Backend support](backends.md)
- [Benchmark methodology](benchmarking.md)
- [Release process](releasing.md)
