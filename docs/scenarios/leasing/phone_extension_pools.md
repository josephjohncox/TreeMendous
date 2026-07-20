# Phone extension pools

`PhoneExtensionPool` builds one numeric `LeasePool` from an explicit inclusive numbering plan. Emergency numbers are removed as individual reservations, and service ranges are removed as inclusive intervals. This allows a PBX policy to protect local emergency aliases, operator codes, voicemail blocks, and other non-user services before provisioning begins.

```python
extensions = PhoneExtensionPool(
    "headquarters",
    first_extension=1000,
    last_extension=9999,
    emergency_numbers=(112, 911),
    service_ranges=((1200, 1299), (9000, 9099)),
    clock=clock,
)
lease = extensions.acquire(
    "provisioner", ttl=60, count=5,
    start_extension=2000, request_id="sales-team",
)
```

The allocator returns the earliest contiguous allowed block unless `start_extension` is supplied. It never jumps an exact request around a reserved number: a block that touches emergency, service, active, or out-of-plan space fails with `ExtensionUnavailableError`. Owner/request retries return the original record; a changed owner, TTL, count, or exact span conflicts. Renewal requires the current revision. Release and clock expiry return extensions while retaining terminal audit records and issuing a higher token on reuse.

Snapshots expose allowed/available spans, complete lease history, and capacity counters. Checkpoints add the stable `plan_id` and preserve request/token state. Restoring verifies that the checkpoint's sole pool scope matches the plan and assigns a new process-local lineage.

`validate_fence(lease, extension)` uses `("phone-extension-pools", plan_id, extension)`. The plan prevents accidental collision between separate PBX number spaces, while the individual extension remains stable across differently shaped team blocks. Membership is checked before the downstream high-water mark is touched.

This allocator does not dial, update a PBX, or replace a durable provisioning database. It is synchronized only for callers sharing one process. Expiry cannot undo configuration already pushed by a stale provisioner. The protected PBX or provisioning database must durably persist the highest accepted token at the same operation that changes an extension. The sample validator demonstrates rejection semantics and is excluded from checkpoints. Restoring allocation state requires an externally coordinated single-writer takeover and a clock compatible with the checkpoint timestamp.

Run the executable example with:

```console
uv run python examples/applications/leasing/phone_extension_pools.py
```
