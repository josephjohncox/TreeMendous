# Software license seats

`SoftwareSeatPool` creates an independent numbered `LeasePool` for each product. Product capacities are explicit, and an optional entitlement map limits each owner to a maximum number of concurrently active seats for that product. With no entitlement map, any owner may check out capacity. With a map—even an empty one—an absent owner/product grant is denied.

```python
seats = SoftwareSeatPool(
    {"editor": 25, "renderer": 8},
    entitlements={"team-a": {"editor": 5, "renderer": 2}},
    clock=clock,
)
checkout = seats.checkout(
    "editor", "team-a", ttl=30, count=3, request_id="session-9",
)
checkout = seats.renew(checkout, ttl=30)
seats.release(checkout)
```

`checkout` (also available as `acquire`) counts the owner's currently active seats before allocation. It rejects an over-entitlement request even when unused product capacity remains. The engine then allocates the earliest contiguous numbered seats. Product scopes are independent, so seat 1 for one product has no relationship to seat 1 for another.

Identical owner/request retries return the original checkout without consuming entitlement twice. LeasePool rejects changed TTL, owner, count, or allocation arguments. Renewal returns a new revision, and old revisions cannot release the session. Release and elapsed TTL return seats to product capacity. Snapshots and diagnostics expose each product separately. Checkpoints preserve product capacities, including the distinction between unrestricted and explicitly empty entitlement policy, as well as lease/request history.

`validate_fence(checkout, seat_id)` uses `("software-license-seats", product, seat_id)`. This stable key is independent of client session IDs and allocation block shape. When a seat is recycled, its newer token can be accepted and the older checkout is subsequently rejected by the same local high-water mark.

The implementation is an embeddable process-local checkout engine, not a distributed license server. Its lock does not coordinate multiple processes, TTL expiry cannot terminate software already running, and the sample validator cannot protect a remote product. Production integration must persist entitlement decisions, checkout records, and the highest accepted token atomically at the licensed operation. Restoring a checkpoint creates fresh pool lineages after an externally coordinated single-writer takeover; downstream fence state is intentionally not part of that checkpoint.

Run the executable example with:

```console
uv run python examples/applications/leasing/software_license_seats.py
```
