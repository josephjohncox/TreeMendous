# VLAN tag pools

`VlanTagPool` enforces the usable IEEE 802.1Q tag interval 1 through 4094. Values 0 and 4095 never enter a `LeasePool`. Callers can remove additional inclusive ranges globally and can layer different reservations onto each network scope. The same tag may be active in two scopes because each scope has an independent pool and operational meaning.

```python
vlans = VlanTagPool(
    ("campus", "datacenter"),
    reserved_ranges=((1, 9),),
    scope_reserved={"campus": ((100, 199),)},
    clock=clock,
)
lease = vlans.acquire(
    "campus", "controller-a", ttl=30, count=4,
    start_tag=200, request_id="segment-71",
)
```

Allocation is deterministic and contiguous. An exact request outside 1–4094 fails as invalid policy input; an in-range request touching a configured reservation or active lease fails with `VlanUnavailableError`. Unknown scopes fail with `VlanScopeError`. Owner/request retries are idempotent within their scope, while changed normalized arguments conflict. Renewal returns a new revision of the same token, and release or TTL expiry restores capacity. Snapshots, checkpoints, and diagnostics report every scope independently.

`validate_fence(lease, tag)` protects `("vlan-tag-pools", network_scope, tag)`. A network scope is mandatory in the key because tag numbers are intentionally reusable across isolated networks. Within one scope, the per-tag key remains stable even if a controller later allocates a differently shaped contiguous block.

This module allocates numbers; it does not configure switches, verify trunk topology, or establish a distributed controller quorum. Its locks, clock observation, and `FenceValidator` state exist in one process. TTL expiry cannot remove a stale VLAN from a device. A controller integration must atomically store the accepted token with the switch/network operation and make devices or a durable control plane reject older tokens. A checkpoint preserves allocation history but not those downstream marks. `from_checkpoint` creates new lineages, so operators must ensure that the source allocator no longer writes and that the supplied clock does not precede checkpoint time.

Run the executable example with:

```console
uv run python examples/applications/leasing/vlan_tag_pools.py
```
