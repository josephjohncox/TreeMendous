"""Web-crawl engine contracts."""

import pytest

from tests.oracles.applications.partitioning.web_crawl import expected_order
from treemendous.applications.partitioning.web_crawl import (
    CrawlPage,
    WebCrawlEngine,
    normalize_url,
)


def _failed_fetch(_: str) -> CrawlPage:
    raise OSError("down")


def test_normalized_frontier_uses_injected_fetcher_without_network() -> None:
    root = "https://example.test/"
    pages = {
        root: CrawlPage(b"root", ("/b", "/a#frag", "HTTPS://EXAMPLE.TEST:443/a")),
        f"{root}a": CrawlPage(b"a", ("/b",)),
        f"{root}b": CrawlPage(b"b", ()),
    }
    engine = WebCrawlEngine(("HTTPS://EXAMPLE.TEST:443/#x",), pages.__getitem__, max_pages=5)
    snapshot = engine.run()
    links = {url: tuple(normalize_url(link, base=url) for link in page.links) for url, page in pages.items()}
    expected = (root, f"{root}a", f"{root}b")
    empty: tuple[str, ...] = ()
    assert snapshot.visited == expected_order(root, links, 5) == expected
    assert snapshot.frontier == empty


def test_fetch_failure_preserves_frontier_for_retry() -> None:
    engine = WebCrawlEngine(
        ("https://example.test",), _failed_fetch, max_pages=1
    )
    with pytest.raises(RuntimeError, match="fetch failed"):
        engine.crawl_next()
    expected = ("https://example.test/",)
    assert engine.snapshot().frontier == expected
