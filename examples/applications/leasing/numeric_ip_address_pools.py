"""Lease a reserved-aware contiguous IPv4 address block."""

from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.numeric_ip_pools import NumericIPAddressPool


def main() -> None:
    """Run a deterministic address lease lifecycle."""
    clock = LogicalClock()
    addresses = NumericIPAddressPool("192.0.2.0/29", clock=clock)
    lease = addresses.acquire("dhcp-server", ttl=5, count=2, request_id="host-7")
    renewed = addresses.renew(lease, ttl=10)
    assert addresses.validate_fence(renewed, addresses.first_address(renewed))
    print(
        f"leased {addresses.first_address(renewed)} through "
        f"{addresses.last_address(renewed)}"
    )


if __name__ == "__main__":
    main()
