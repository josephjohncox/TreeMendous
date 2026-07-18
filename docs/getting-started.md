# Getting started

Tree-Mendous is alpha software for integer range availability. Install Python
3.11–3.13 and then run `pip install treemendous` or `uv add treemendous`.

## Model

A `Span(start, end)` is half-open: `[start, end)`. Adjacent spans do not overlap.
Coordinates must be integers (booleans are rejected), and `start < end`.
Canonical range sets have an explicit `ManagedDomain`, which may contain
multiple disjoint spans. Gaps between those spans are outside the managed space,
not occupied space.

Start with the [README quickstart](../README.md#quickstart) or run the maintained
example:

```bash
python examples/basic_rangeset.py
```

The default factory probes installed backends and selects a stable 64-bit
implementation. Pass `backend="py_boundary"` for a predictable pure-Python
installation. `allocate` holds one lock from first-fit search through removal;
it returns `None` without mutation when no fit exists.

## Next steps

- [API and migration](api.md)
- [Backend selection and maturity](backends.md)
- [Native and accelerator builds](building.md)
