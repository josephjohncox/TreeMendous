"""Correctness-checked offline smoke workload for web crawling."""

from tests.oracles.applications.partitioning.web_crawl import expected_order
from treemendous.applications.partitioning.web_crawl import CrawlPage, WebCrawlEngine


def run_smoke() -> int:
    pages = {
        f"https://crawl.test/{i}": CrawlPage(
            str(i).encode(),
            tuple(f"https://crawl.test/{child}" for child in (i + 1, i + 2) if child < 100),
        )
        for i in range(100)
    }
    engine = WebCrawlEngine(("https://crawl.test/0",), pages.__getitem__, max_pages=100)
    observed = engine.run().visited
    links = {url: page.links for url, page in pages.items()}
    if observed != expected_order("https://crawl.test/0", links, 100):
        raise AssertionError("crawl smoke differs from frontier oracle")
    return len(observed)
