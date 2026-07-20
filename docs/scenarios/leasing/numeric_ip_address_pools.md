# Numeric IP address pools

`NumericIPAddressPool` accepts one canonical IPv4 or IPv6 CIDR and encodes every address with Python's standard `ipaddress` integer representation. The resulting `LeasePool` spans are half-open integer ranges, which makes contiguous address allocation deterministic without adding a dependency. A request can choose the earliest block or name an exact `start_address`; a block that crosses the CIDR boundary is rejected before mutation.

The network address is reserved by default. IPv4 broadcast is reserved by default, while IPv6 does not reserve its last address unless `reserve_broadcast=True` is explicit. `reserved` removes additional individual addresses. A `/31`, `/32`, or similarly small range can therefore become empty under the selected policy, and construction reports that honestly rather than creating an unusable engine.

```python
pool = NumericIPAddressPool(
    "192.0.2.0/24", reserved=("192.0.2.10",), clock=clock,
)
lease = pool.acquire(
    "dhcp", ttl=60, count=8,
    start_address="192.0.2.32", request_id="offer-7",
)
print(pool.first_address(lease), pool.last_address(lease))
```

Owner/request idempotency, TTL renewal, release, expiry, terminal history, diagnostics, and token ordering come from `LeasePool`. Domain failures distinguish an invalid/out-of-CIDR address from a valid but reserved or busy block. Snapshots expose encoded spans. Checkpoints include the CIDR and pool state; restoring checks that the canonical scope still matches and starts a fresh local lineage.

`validate_fence(handle, address)` first validates the address family, CIDR membership, and lease membership. It then uses the stable key `("numeric-ip-address-pools", canonical_cidr, encoded_address)`. This per-address key prevents a later overlapping allocation with different block boundaries from escaping the high-water mark.

Neither leasing nor fencing is distributed. The built-in lock protects threads in one process only. A DHCP server, IPAM database, or router must persist assignment and fencing token atomically. Expiry merely makes capacity eligible for reassignment; it cannot revoke configuration already applied by a stale client. A checkpoint also excludes the sample downstream validator, so operational takeover must retire the source process and restore any real downstream high-water state separately.

Run the executable example with:

```console
uv run python examples/applications/leasing/numeric_ip_address_pools.py
```
