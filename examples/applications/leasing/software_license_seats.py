"""Checkout and renew an entitled product seat."""

from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.software_seats import SoftwareSeatPool


def main() -> None:
    """Run an entitled software checkout lifecycle."""
    clock = LogicalClock()
    seats = SoftwareSeatPool(
        {"editor": 4},
        entitlements={"alice": {"editor": 2}},
        clock=clock,
    )
    checkout = seats.checkout("editor", "alice", ttl=2, count=2)
    clock.advance()
    checkout = seats.renew(checkout, ttl=4)
    assert seats.validate_fence(checkout, checkout.resource.start)
    print(f"editor seats {checkout.resource}, expires {checkout.expires_at}")


if __name__ == "__main__":
    main()
