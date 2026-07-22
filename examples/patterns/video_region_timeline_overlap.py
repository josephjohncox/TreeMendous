"""Query experimental process-local video regions over a timeline.

``BoxIndex3D`` treats x, y, and time as half-open integer axes. It supplies no
frame-rate/timebase conversion, pixel/codec semantics, polygon masks, track
identity, media persistence, edit history, or distributed collaboration.
"""

from treemendous.multidimensional import Box, BoxIndex3D


def main() -> None:
    regions = BoxIndex3D()
    duplicate = Box((10, 10, 1_000), (30, 30, 1_100))
    title = regions.insert(duplicate, "title")
    duplicate_title = regions.insert(duplicate, "title-copy")
    later = regions.insert(Box((10, 10, 1_100), (30, 30, 1_200)), "later")

    query = Box((20, 20, 1_050), (21, 21, 1_060))
    matches = regions.overlaps(query)
    assert tuple(entry.handle for entry in matches) == (title, duplicate_title)
    assert tuple(entry.data for entry in matches) == ("title", "title-copy")
    assert regions.overlaps(Box((20, 20, 1_100), (21, 21, 1_101))) == (
        regions.get(later),
    )
    regions.update(duplicate_title, data="title-copy-reviewed")
    assert regions.remove(title).data == "title"
    assert tuple(entry.data for entry in regions.entries()) == (
        "title-copy-reviewed",
        "later",
    )
    print("matches=title,title-copy order=1,2 touching=later remaining=2,3")


if __name__ == "__main__":
    main()
