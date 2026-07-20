"""Compute track-specific render invalidation coverage."""

from treemendous.applications.catalogs.video_edit_regions import VideoEditCatalog


def main() -> None:
    catalog = VideoEditCatalog()
    catalog.add("grade", 0, 120, track="video-1", effect="color")
    catalog.add("title", 60, 180, track="video-1", effect="overlay")
    result = catalog.invalidation(50, 130, tracks=frozenset({"video-1"}))
    print(result.coverage.segments)


if __name__ == "__main__":
    main()
