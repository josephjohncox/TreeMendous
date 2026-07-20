# Database key-range locks

## Model

`DatabaseKeyRangeLocks` applies shared/exclusive transaction locking to table-local key bands. Each active lock retains its transaction owner, original start and end keys, encoded integer endpoints, mode, handle, and acquisition order. Locks on different tables are independent; coincident locks are never coalesced.

`encode_key` UTF-8 encodes strings (or accepts bytes), converts each byte to a nonzero base-257 digit, and pads to a fixed 32-byte width. The result is deterministic and preserves bytewise ordering, including prefix ordering. Keys over 32 bytes and empty keys are rejected, making the encoding's domain explicit rather than relying on process-randomized hashes.

## Lock lifecycle

A range requires `start_key < end_key` after encoding. Shared/shared overlaps are compatible; any overlap with an exclusive mode conflicts across owners. `conflicts` is read-only. `upgrade` atomically changes a shared lock to exclusive when possible. `release` requires the owning transaction and removes one handle only. `snapshot` provides deterministic table-name and acquisition ordering.

## Example

```python
handle = locks.acquire("accounts", "tx-17", "a", "m", "shared")
conflicts = locks.conflicts("accounts", "tx-18", "c", "d", "exclusive")
locks.release(handle, owner="tx-17")
```

The encoding models bounded bytewise keys, not locale collation or a database's custom comparator.
