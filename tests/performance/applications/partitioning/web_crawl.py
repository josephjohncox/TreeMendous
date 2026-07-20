"""Attested offline benchmark for deterministic web crawling."""

from __future__ import annotations

from tests.oracles.applications.partitioning.web_crawl import (
    expected_frontier,
    expected_order,
)
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from tests.performance.applications.partitioning._common import validate_case
from treemendous.applications.partitioning.web_crawl import (
    CrawlPage,
    CrawlSnapshot,
    WebCrawlEngine,
)

_DEFAULT_OPERATIONS = 120
_MAX_OPERATIONS = 750
_DEFAULT_SEED = 59


def run_benchmark(
    operations: int = _DEFAULT_OPERATIONS, seed: int = _DEFAULT_SEED
) -> ApplicationSample:
    """Crawl and attest a bounded in-memory page graph without network I/O."""
    validate_case(operations, seed, maximum=_MAX_OPERATIONS)
    urls = tuple(f"https://crawl.test/page-{index:04}" for index in range(operations))
    pages = {
        url: CrawlPage(
            f"body-{index:04}-{seed}".encode(),
            tuple(
                urls[child]
                for child in (index + 1, index + 2, index + 5 + seed % 3)
                if child < operations
            ),
        )
        for index, url in enumerate(urls)
    }
    links = {url: page.links for url, page in pages.items()}
    engine = WebCrawlEngine((urls[0],), pages.__getitem__, max_pages=operations)

    def execute() -> CrawlSnapshot:
        return engine.run()

    def observe(raw: CrawlSnapshot) -> ApplicationOutcome:
        snapshot = engine.snapshot()
        return ApplicationOutcome(
            results=raw.visited,
            final_state={
                "visited": snapshot.visited,
                "frontier": snapshot.frontier,
                "pages": snapshot.pages,
            },
            counters={
                "pages_fetched": len(snapshot.visited),
                "bodies_stored": len(snapshot.pages),
                "run_calls": 1,
            },
        )

    def oracle() -> ApplicationOutcome:
        visited = expected_order(urls[0], links, operations)
        frontier = expected_frontier(urls[0], links, operations)
        stored_pages = tuple((url, pages[url].body) for url in sorted(visited))
        return ApplicationOutcome(
            results=visited,
            final_state={
                "visited": visited,
                "frontier": frontier,
                "pages": stored_pages,
            },
            counters={
                "pages_fetched": len(visited),
                "bodies_stored": len(stored_pages),
                "run_calls": 1,
            },
        )

    return run_application_case(
        scenario_id="partitioning.web_crawl",
        operations=operations,
        execute=execute,
        observe=observe,
        oracle=oracle,
    )


def run_smoke() -> int:
    """Run the default attested workload and return its operation count."""
    return run_benchmark().operations
