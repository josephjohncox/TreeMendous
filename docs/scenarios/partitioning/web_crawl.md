# Distributed web crawl

`WebCrawlEngine` is an offline-capable crawl state machine. `normalize_url()` accepts absolute HTTP(S), lowercases scheme/host, removes default ports and fragments, normalizes paths, sorts query pairs, and rejects credentials/invalid ports. Seeds are normalized and deduplicated. The FIFO frontier visits each known URL once; each fetched page's links are resolved relative to its URL, normalized, deduplicated, sorted, then appended.

The fetcher is injected and must return `CrawlPage(body: bytes, links: tuple[str, ...])`; the engine never opens a socket implicitly. Fetch failure restores the URL to the front, abandons the claim, and propagates a contextual error. `run()` obeys `max_pages`; snapshots/checkpoints expose visited order, frontier, bodies, and local claims/events.

Network policy remains the caller's responsibility: robots.txt, rate limiting, redirects, content limits, DNS safety and authentication are deliberately absent. Distributed crawling additionally requires a durable normalized visited set, frontier transactions and fenced page commits.

The example uses an in-memory fetch map and runs from any directory. The smoke crawls 100 fixture pages and compares traversal with an independent finite-frontier oracle.
