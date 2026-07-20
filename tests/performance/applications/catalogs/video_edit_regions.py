"""Smoke benchmark for video render invalidation queries."""

from time import perf_counter

from treemendous.applications.catalogs.video_edit_regions import VideoEditCatalog


def run_smoke(iterations: int = 500) -> float:
    catalog = VideoEditCatalog()
    for index in range(200):
        catalog.add(
            f"r{index}",
            index * 5,
            index * 5 + 30,
            track=f"v{index % 4}",
            effect="grade",
        )
    started = perf_counter()
    for index in range(iterations):
        start = index % 900
        catalog.invalidation(start, start + 50, tracks=frozenset({"v1"}))
    return perf_counter() - started


if __name__ == "__main__":
    print(run_smoke())
