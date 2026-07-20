#!/usr/bin/env python3
"""Run an offline injected-fetcher crawl from any working directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from treemendous.applications.partitioning.web_crawl import CrawlPage, WebCrawlEngine


def main() -> None:
    pages = {
        "https://example.test/": CrawlPage(b"root", ("/a", "/a#duplicate")),
        "https://example.test/a": CrawlPage(b"a", ()),
    }
    snapshot = WebCrawlEngine(("https://example.test",), pages.__getitem__, max_pages=3).run()
    if snapshot.visited != ("https://example.test/", "https://example.test/a"):
        raise RuntimeError("unexpected crawl order")
    print("web-crawl: visited 2 normalized URLs offline")


if __name__ == "__main__":
    main()
