# TCP and UDP port leases

`PortLeaseEngine` allocates contiguous half-open port spans while presenting ports as the familiar inclusive integers 1 through 65535. TCP and UDP use separate `LeasePool` instances, so TCP 8080 and UDP 8080 can be held concurrently. By default, privileged/system ports 1–1023 and the dynamic/ephemeral interval 49152–65535 are excluded. `protocol_reserved` adds transport-specific inclusive exclusions before either pool is created. Exact requests that touch any exclusion fail rather than being silently moved.

```python
lease = ports.acquire(
    "tcp", "api-server", ttl=30, count=4,
    start_port=8000, request_id="deploy-19",
)
lease = ports.renew(lease, ttl=30)
ports.release(lease)
```

A repeated `request_id` with the same owner, TTL, size, protocol, and exact span returns the original record. A changed retry raises the shared request-conflict failure. Expiry is materialized by `expire`, `snapshot`, `checkpoint`, `diagnostics`, or the next mutation. Released and expired records remain visible, while their ports become allocatable with a higher token.

`validate_fence(handle, port)` uses the stable key `("tcp-udp-port-leases", protocol, port)`. The key is per port, not per requested block, so differently shaped later blocks cannot bypass a prior high-water mark. Validation requires that the port lie inside the handle. Equal-token retries are accepted only with the same lease identity by the underlying `FenceValidator`.

The allocator, clock checks, lock, and sample fence validator are process-local. Expiry does not stop an old service, and validation does not become durable merely because the engine returned `True`. A real socket broker or service supervisor must atomically persist and reject stale tokens at the protected operation. `checkpoint` captures lease history and counters but intentionally excludes downstream fence state. `from_checkpoint` creates new pool lineages; callers must retire the old writer and supply a clock in a compatible epoch.

Run the executable lifecycle with:

```console
uv run python examples/applications/leasing/tcp_udp_port_leases.py
```
