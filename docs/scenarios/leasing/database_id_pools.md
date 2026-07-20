# Database ID pools

`DatabaseIdPool` makes the distinction between temporary authority and permanent identity explicit. A normal `acquire` reserves a contiguous batch at `next_monotonic_id` and advances that cursor. Expiring or releasing the batch does **not** move the monotonic cursor backward. Instead, uncommitted IDs enter a separate reusable queue. They can only be issued again when the caller passes `reusable=True`.

```python
batch = ids.acquire("writer", ttl=20, count=100, request_id="txn-41")
committed = ids.commit(batch)          # permanent IDs, no TTL renewal
scratch = ids.acquire("loader", ttl=5, count=10)
ids.release(scratch)
retry = ids.acquire("loader-2", ttl=5, count=10, reusable=True)
```

`commit` terminates the source lease and records a `CommittedIdBatch` that is neither renewable nor releasable. Its IDs never enter the reusable queue. This is deliberately different from `release` and clock expiry, both of which mark uncommitted capacity reusable. Normal acquisitions remain monotonic even when recycled space exists. Exhausting the configured maximum produces `DatabaseIdUnavailableError`; asking for reuse with no sufficiently contiguous recycled span produces the same domain availability category.

Request IDs cover owner, TTL, count, reuse mode, and the selected exact span. Identical retries return the original active or terminal lease. A retry does not allocate another batch or advance the cursor. Committing the same unchanged handle is also idempotent. Snapshots separately report the monotonic cursor, reusable spans, committed batches, and underlying lease history. Checkpoints preserve all four plus request records and the next fencing token.

`validate_fence(handle, identifier)` accepts either a temporary `DatabaseIdLease` or a `CommittedIdBatch`. It uses the stable key `("database-id-pools", namespace, identifier)`, so each protected database ID has one high-water sequence independent of batch shape. The committed object retains its source token as durable write evidence.

This engine is not a database sequence, transaction manager, or consensus service. Its allocation and `commit` transition are atomic only with respect to threads using the same object. Applications must commit business rows and permanent ID evidence in their own durable transaction. Checkpoint restoration creates a new process-local lineage and does not coordinate source shutdown. The sample `FenceValidator` is also local and excluded from checkpoints; a real database must persist its token beside the protected write.

Run the executable example with:

```console
uv run python examples/applications/leasing/database_id_pools.py
```
