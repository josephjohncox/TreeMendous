#!/usr/bin/env python3
"""Run deterministic document search from any working directory."""

from treemendous.applications.partitioning.document_search import DocumentSearchEngine


def main() -> None:
    engine = DocumentSearchEngine(
        {2: "range tree", 1: "range", 3: "tree"}, "range tree"
    )
    hits = engine.run(shard_size=1)
    if tuple(hit.document_id for hit in hits) != (2,):
        raise RuntimeError("unexpected document search result")
    print("document-search: 1 hit")


if __name__ == "__main__":
    main()
