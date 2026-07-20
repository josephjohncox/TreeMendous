"""Lease and fence a TCP port, then recycle it after expiry."""

from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.tcp_udp_ports import PortLeaseEngine


def main() -> None:
    """Run a deterministic port lease lifecycle."""
    clock = LogicalClock()
    ports = PortLeaseEngine(clock=clock)
    old = ports.acquire("tcp", "web-a", ttl=2, request_id="deploy-a")
    assert ports.validate_fence(old, old.resource.start)
    clock.advance(2)
    ports.expire()
    current = ports.acquire("tcp", "web-b", ttl=2)
    assert current.resource == old.resource
    assert ports.validate_fence(current, current.resource.start)
    assert not ports.validate_fence(old, old.resource.start)
    print(f"tcp port {current.resource.start}, fence {current.token}")


if __name__ == "__main__":
    main()
