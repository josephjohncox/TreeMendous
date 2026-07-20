"""Run the experimental guarded bounded sparse-grid box index."""

from treemendous.multidimensional import BoundedBoxIndex, Box


def main() -> None:
    index = BoundedBoxIndex(
        Box((0, 0, 0), (20, 20, 20)),
        (5, 5, 5),
        max_cells_per_entry=32,
        max_cells_per_query=32,
    )
    first = index.insert(Box((1, 1, 1), (7, 7, 7)), "first")
    second = index.insert(Box((5, 5, 5), (9, 9, 9)), "second")
    matches = index.overlaps(Box((6, 6, 6), (8, 8, 8)))
    diagnostics = index.diagnostics()
    print(
        f"matches={len(matches)} handles={first.sequence},{second.sequence} "
        f"grid={diagnostics.grid_shape} postings={diagnostics.posting_count}"
    )


if __name__ == "__main__":
    main()
