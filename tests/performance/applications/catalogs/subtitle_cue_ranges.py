"""Smoke benchmark for active subtitle cue queries."""

from time import perf_counter

from treemendous.applications.catalogs.subtitle_cue_ranges import SubtitleCatalog


def run_smoke(iterations: int = 500) -> float:
    catalog = SubtitleCatalog()
    for index in range(200):
        catalog.add(
            f"c{index}",
            index * 100,
            index * 100 + 500,
            language="en",
            layer=index % 3,
            text="cue",
        )
    started = perf_counter()
    for index in range(iterations):
        catalog.active_at(index * 30 % 20_000, language="en")
    return perf_counter() - started


if __name__ == "__main__":
    print(run_smoke())
