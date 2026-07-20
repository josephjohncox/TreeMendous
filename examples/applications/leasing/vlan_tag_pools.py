"""Lease the same VLAN number in independent network scopes."""

from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.vlan_tags import VlanTagPool


def main() -> None:
    """Run scoped VLAN allocations with a reserved management range."""
    clock = LogicalClock()
    vlans = VlanTagPool(
        ("campus", "datacenter"),
        reserved_ranges=((1, 9),),
        clock=clock,
    )
    campus = vlans.acquire("campus", "controller-a", ttl=5)
    datacenter = vlans.acquire("datacenter", "controller-b", ttl=5)
    assert campus.resource == datacenter.resource
    assert vlans.validate_fence(campus, campus.resource.start)
    print(f"VLAN {campus.resource.start} is independently scoped")


if __name__ == "__main__":
    main()
