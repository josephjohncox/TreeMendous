"""Index experimental process-local warehouse space-time reservations.

``BoxIndex3D`` supplies only half-open axis-aligned integer boxes and local
record identity. It does not supply warehouse units/maps, aisle or vehicle
geometry, routing, authorization, durable reservations, or coordination.
"""

from treemendous.multidimensional import Box, BoxIndex3D


def main() -> None:
    reservations = BoxIndex3D()
    duplicate = Box((0, 0, 100), (4, 4, 120))
    first = reservations.insert(duplicate, "forklift-a")
    second = reservations.insert(duplicate, "forklift-b")
    touching = reservations.insert(Box((4, 0, 100), (8, 4, 120)), "forklift-c")

    query = Box((1, 1, 110), (2, 2, 111))
    assert tuple(entry.data for entry in reservations.overlaps(query)) == (
        "forklift-a",
        "forklift-b",
    )
    assert not duplicate.overlaps(Box((4, 0, 100), (5, 1, 101)))
    removed = reservations.remove(second)
    updated = reservations.update(first, data="forklift-a-confirmed")
    assert removed.data == "forklift-b"
    assert updated.data == "forklift-a-confirmed"
    assert tuple(entry.handle for entry in reservations.entries()) == (first, touching)
    print("conflicts=forklift-a,forklift-b handles=1,2,3 remaining=1,3")


if __name__ == "__main__":
    main()
