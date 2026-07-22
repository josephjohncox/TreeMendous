"""Query experimental, process-local spatiotemporal box records.

This pattern is outside the 50-engine registry. ``BoxIndex3D`` is experimental
and process-local; the integer axes do not supply a geographic coordinate
system, durable storage, distributed queries, or application authorization.
"""

from treemendous.multidimensional import Box, BoxIndex3D


def main() -> None:
    index = BoxIndex3D()
    first = index.insert(Box((0, 0, 100), (10, 10, 120)), "alpha")
    second = index.insert(Box((8, 8, 110), (16, 16, 130)), "beta")
    third = index.insert(Box((20, 20, 100), (30, 30, 130)), "remote")

    observation = Box((9, 9, 115), (12, 12, 116))
    matches = index.overlaps(observation)
    labels = ",".join(entry.data for entry in matches)
    handles = ",".join(str(handle.sequence) for handle in (first, second, third))
    print(f"matches={labels} handles={handles} process_local=True")


if __name__ == "__main__":
    main()
