"""Hand off a region band with a higher fencing token."""

from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.game_regions import GameRegionPool


def main() -> None:
    """Run an idempotent ownership handoff."""
    clock = LogicalClock()
    regions = GameRegionPool({"eu-1": (100, 199)}, clock=clock)
    old = regions.acquire("eu-1", "server-a", ttl=5, count=4)
    assert regions.validate_fence(old, old.resource.start)
    current = regions.handoff(old, "server-b", ttl=5, request_id="move-42")
    assert current.resource == old.resource
    assert regions.validate_fence(current, current.resource.start)
    assert not regions.validate_fence(old, old.resource.start)
    print(f"handoff {current.resource} to {current.owner}, fence {current.token}")


if __name__ == "__main__":
    main()
