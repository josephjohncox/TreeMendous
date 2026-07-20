"""Run the fixed-dimensional experimental axis-projection indexes."""

from treemendous.multidimensional import Box, BoxIndex2D, BoxIndex3D, BoxIndex4D


def main() -> None:
    for index_type in (BoxIndex2D, BoxIndex3D, BoxIndex4D):
        index = index_type()
        dimensions = index.dimensions
        duplicate = Box((0,) * dimensions, (4,) * dimensions)
        first = index.insert(duplicate, "first")
        second = index.insert(duplicate, "second")
        index.insert(Box((10,) * dimensions, (12,) * dimensions), "far")
        matches = index.overlaps(Box((2,) * dimensions, (3,) * dimensions))
        print(
            f"{index_type.__name__}: matches={len(matches)} "
            f"handles={first.sequence},{second.sequence} "
            f"algorithm={index.diagnostics().algorithm}"
        )


if __name__ == "__main__":
    main()
