# Game world region IDs

`GameRegionPool` treats each shard as an independent numeric region namespace. A server can acquire an earliest contiguous band, name an exact `start_region`, or request a band adjacent to an active band it already owns. An adjacency anchor must be current, on the same shard, and owned by the same server. The engine tries the numeric band immediately after the anchor, or the band before it when the shard boundary prevents forward growth.

```python
owned = regions.acquire("eu-1", "server-a", ttl=15, count=8)
expanded = regions.acquire(
    "eu-1", "server-a", ttl=15, count=4,
    adjacent_to=owned, request_id="expand-2",
)
current = regions.handoff(
    owned, "server-b", ttl=15, request_id="move-42",
)
```

`handoff` serializes release of the source band and exact reacquisition by the new owner in this process. Reacquisition receives a higher shard-local fencing token. Handoff request IDs are retained with their source token, shard, new owner, TTL, and result token, making identical transfer retries return the same result. Changed retries fail rather than transferring another band.

Ordinary acquisitions have LeasePool owner/request idempotency, renewal revisions, explicit release, and clock expiry. Snapshots and diagnostics are grouped by shard. Checkpoints preserve shard bounds, ordinary request history, transfer records, terminal leases, and token counters. Restoring assigns new pool lineage IDs, making source-process handles foreign.

`validate_fence(lease, region_id)` uses `("game-world-region-ids", shard, region_id)`. After accepting the transferred lease at a region, the old owner's lower token is rejected. The key is per region rather than per band, so a later split or merge cannot evade the high-water mark.

The crucial boundary is that this is not distributed ownership. The handoff lock is process-local and does not atomically update a game database, routing layer, or two independent servers. Expiry and release cannot stop an old simulation loop. A production shard directory must atomically persist the owner and highest token, route work using that state, and reject stale writes. The downstream validator included here demonstrates token semantics only and is excluded from checkpoints. Exclusive source retirement and destination startup around `from_checkpoint` are external operational steps.

Run the executable handoff with:

```console
uv run python examples/applications/leasing/game_world_region_ids.py
```
