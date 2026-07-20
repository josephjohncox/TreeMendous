"""Lease bins only from a compatible zone."""

from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.warehouse_bins import BinZone, WarehouseBinPool


def main() -> None:
    """Run a compatible warehouse range lifecycle."""
    clock = LogicalClock()
    bins = WarehouseBinPool(
        {
            "chemical": BinZone(
                500,
                520,
                size_classes=frozenset({"large"}),
                hazards=frozenset({"flammable"}),
            )
        },
        clock=clock,
    )
    lease = bins.acquire(
        "chemical",
        "putaway-17",
        ttl=5,
        count=3,
        size_class="large",
        hazard="flammable",
    )
    assert bins.validate_fence(lease, lease.resource.start)
    print(f"assigned compatible bins {lease.resource}")


if __name__ == "__main__":
    main()
