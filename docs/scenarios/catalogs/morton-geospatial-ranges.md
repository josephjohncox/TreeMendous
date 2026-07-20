# Morton-code geospatial ranges

## Approximate model

`MortonGeospatialCatalog` is explicitly an approximate one-dimensional candidate index, never an exact Cartesian index. `morton_encode` interleaves bits from bounded nonnegative integer `x` and `y` coordinates; `morton_decompose` reverses that operation. Bit width is explicit and validated.

Each retained item keeps its exact half-open Cartesian rectangle and label. Its production interval is the Morton band from the lower corner through the upper included corner. Monotonic interleaving makes this band a candidate superset, but it commonly contains codes outside the rectangle. `approximate_candidates` exposes that broad phase honestly. `search` always applies rectangle-intersection tests to every candidate and removes false positives before returning records in insertion order.

## Mutation and validation

Rectangles must be nonempty and fit inside `[0, 2**bits)` on both axes. IDs and labels are nonempty. `update` changes bounds or label without replacing the handle; `remove` deletes one identity. `snapshot` retains the Cartesian payload and approximate Morton interval for independent inspection.

## Example

```python
catalog = MortonGeospatialCatalog(bits=16)
catalog.add("park", 10, 10, 30, 25, label="park")
exact_hits = catalog.search(20, 20, 40, 40)
```

The design avoids dependency on a multidimensional backend while remaining honest about candidate-band false positives and exact filtering cost.
