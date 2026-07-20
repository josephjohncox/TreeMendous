#!/usr/bin/env python3
"""Run term-band posting-list merge from any working directory."""

from treemendous.applications.partitioning.index_merge import IndexMergeEngine


def main() -> None:
    merged = IndexMergeEngine(({"a": (1, 3)}, {"a": (2, 3), "b": (4,)})).run(band_size=1)
    observed = tuple((item.term, item.postings) for item in merged)
    if observed != (("a", (1, 2, 3)), ("b", (4,))):
        raise RuntimeError("unexpected index merge")
    print("index-merge: merged 2 term bands")


if __name__ == "__main__":
    main()
