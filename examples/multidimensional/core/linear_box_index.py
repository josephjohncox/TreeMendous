"""Demonstrate duplicate identity in the experimental linear BoxIndex."""

from treemendous.multidimensional import Box, BoxIndex


def main() -> None:
    index = BoxIndex(2)
    tile = Box((0, 0), (8, 8))
    first = index.insert(tile, "primary")
    second = index.insert(tile, "secondary")
    matches = index.overlaps(Box((4, 4), (5, 5)))

    updated = index.update(first, data="primary-updated")
    removed = index.remove(second)

    print(
        f"matches={len(matches)} handles={first.sequence},{second.sequence} "
        f"updated={updated.data} removed={removed.data} remaining={len(index)}"
    )


if __name__ == "__main__":
    main()
