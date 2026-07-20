"""Commit one ID batch and explicitly recycle another."""

from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.leasing.database_ids import DatabaseIdPool


def main() -> None:
    """Run permanent and reusable database ID paths."""
    clock = LogicalClock()
    ids = DatabaseIdPool("orders", maximum_id=100, clock=clock)
    permanent = ids.commit(ids.acquire("writer", ttl=5, count=3))
    temporary = ids.acquire("import", ttl=1, count=2)
    clock.advance()
    ids.expire()
    recycled = ids.acquire("retry", ttl=3, count=2, reusable=True)
    assert recycled.resource == temporary.resource
    assert ids.validate_fence(permanent, permanent.resource.start)
    print(f"committed {permanent.resource}; recycled {recycled.resource}")


if __name__ == "__main__":
    main()
