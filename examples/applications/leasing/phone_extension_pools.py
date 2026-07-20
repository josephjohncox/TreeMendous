"""Lease extensions around emergency and service reservations."""

from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.phone_extensions import PhoneExtensionPool


def main() -> None:
    """Run a reserved numbering-plan assignment."""
    clock = LogicalClock()
    extensions = PhoneExtensionPool(
        "hq",
        first_extension=100,
        last_extension=199,
        emergency_numbers=(112,),
        service_ranges=((120, 129),),
        clock=clock,
    )
    lease = extensions.acquire("pbx", ttl=5, count=3, request_id="team-sales")
    assert 112 not in range(lease.resource.start, lease.resource.end)
    assert extensions.validate_fence(lease, lease.resource.start)
    print(f"assigned extensions {lease.resource}")


if __name__ == "__main__":
    main()
