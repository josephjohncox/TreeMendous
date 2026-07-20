# Examples

## Stable one-dimensional API

[`basic_rangeset.py`](basic_rangeset.py) uses only the stable public
`RangeSet` API:

```bash
uv run python examples/basic_rangeset.py
```

## Experimental multidimensional API

[`multidimensional/core/linear_box_index.py`](multidimensional/core/linear_box_index.py)
demonstrates duplicate record identity, deterministic overlap order, update,
and exact handle removal:

```bash
uv run python examples/multidimensional/core/linear_box_index.py
```

Backend implementation modules remain internal. Applications should construct
one-dimensional range sets through `treemendous.create_range_set` and import
experimental multidimensional values explicitly from
`treemendous.multidimensional`.
