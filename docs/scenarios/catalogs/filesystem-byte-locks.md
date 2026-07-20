# Filesystem byte-range locks

## Model

`FilesystemByteLocks` owns an independent range-lock table for each file. Every acquisition has a stable handle, owner, half-open byte range, mode, and acquisition order. Shared locks from different owners may overlap. An exclusive lock conflicts with every overlapping other-owner lock, while an owner's own locks are reentrant and remain distinct identities.

`conflicts` returns every incompatible lock in acquisition order without modifying state. `active` returns all identities intersecting a file range. File qualification prevents numerically identical ranges on different files from conflicting.

## Ownership and lifecycle

Release requires both the lock handle and an explicit owner assertion. A reconstructed or stolen handle is not authorization; a wrong owner raises `PermissionError`. Duplicate acquisitions require duplicate releases. `upgrade` is the domain update operation: it atomically changes shared to exclusive only when no other-owner lock conflicts. `snapshot` sorts files by name and preserves per-file acquisition order.

## Example

```python
first = locks.acquire("data.bin", "reader-a", 0, 4096, "shared")
locks.acquire("data.bin", "reader-b", 0, 4096, "shared")
locks.release(first, owner="reader-a")
```

Ranges must be valid nonempty half-open integer spans. This engine models advisory lock compatibility and mandatory ownership for engine mutation; operating-system enforcement is outside its scope.
