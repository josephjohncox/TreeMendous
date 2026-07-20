"""Deterministic URL-normalizing crawl frontier with an injected fetcher."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from posixpath import normpath
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from treemendous.applications._shared.claiming import ClaimUnavailableError
from treemendous.applications._shared.clock import Clock
from treemendous.applications.partitioning._runtime import PartitionRuntime, positive


@dataclass(frozen=True)
class CrawlPage:
    """Fetcher output: body bytes and discovered links."""

    body: bytes
    links: tuple[str, ...]


Fetcher = Callable[[str], CrawlPage]


@dataclass(frozen=True)
class CrawlSnapshot:
    """Immutable visited order, pending frontier, and fetched bodies."""

    visited: tuple[str, ...]
    frontier: tuple[str, ...]
    pages: tuple[tuple[str, bytes], ...]


def normalize_url(url: str, *, base: str | None = None) -> str:
    """Return a fragment-free canonical HTTP(S) URL."""
    if not isinstance(url, str) or not url:
        raise ValueError("URL must be a nonempty string")
    combined = urljoin(base, url) if base is not None else url
    parsed = urlsplit(combined)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError("URL must be absolute HTTP or HTTPS")
    host = parsed.hostname.lower()
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("URL port is invalid") from exc
    default = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    netloc = host if port is None or default else f"{host}:{port}"
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL user information is not supported")
    path = normpath(parsed.path or "/")
    if not path.startswith("/"):
        path = f"/{path}"
    if parsed.path.endswith("/") and not path.endswith("/"):
        path += "/"
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


class WebCrawlEngine:
    """Visit a bounded deterministic frontier using a caller-provided fetcher.

    No network implementation is included or invoked implicitly. Claims,
    frontier, and visited state are process-local; a distributed crawler needs
    a durable normalized URL set and fencing-token-protected page commits.
    """

    def __init__(
        self,
        seeds: Sequence[str],
        fetcher: Fetcher,
        *,
        max_pages: int,
        clock: Clock | None = None,
    ) -> None:
        if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence):
            raise TypeError("seeds must be a sequence")
        if not seeds:
            raise ValueError("seeds must not be empty")
        if not callable(fetcher):
            raise TypeError("fetcher must be callable")
        positive(max_pages, "max_pages")
        normalized = tuple(sorted({normalize_url(seed) for seed in seeds}))
        self._fetcher = fetcher
        self._max_pages = max_pages
        self._frontier: deque[str] = deque(normalized)
        self._known = set(normalized)
        self._visited: list[str] = []
        self._pages: dict[str, bytes] = {}
        self._runtime = PartitionRuntime(max_pages, clock=clock)

    def crawl_next(self, *, owner: str = "local") -> str | None:
        """Claim one fetch ordinal, fetch a URL, and merge normalized links."""
        if not self._frontier or len(self._visited) >= self._max_pages:
            return None
        claim = self._runtime.claim(owner, 1)
        url = self._frontier.popleft()
        try:
            page = self._fetcher(url)
            if not isinstance(page, CrawlPage):
                raise TypeError("fetcher must return CrawlPage")
            if not isinstance(page.body, bytes):
                raise TypeError("page body must be bytes")
            links = tuple(sorted({normalize_url(link, base=url) for link in page.links}))
        except (Exception,) as exc:
            self._frontier.appendleft(url)
            self._runtime.abandon(claim)
            raise RuntimeError(f"fetch failed for {url}") from exc
        self._visited.append(url)
        self._pages[url] = page.body
        for link in links:
            if link not in self._known:
                self._known.add(link)
                self._frontier.append(link)
        self._runtime.complete(claim, "fetched", {"links": len(links)})
        return url

    def run(self) -> CrawlSnapshot:
        """Crawl until the frontier or page budget is exhausted."""
        while self._frontier and len(self._visited) < self._max_pages:
            try:
                self.crawl_next()
            except ClaimUnavailableError:
                break
        return self.snapshot()

    def snapshot(self) -> CrawlSnapshot:
        """Return detached crawl state."""
        return CrawlSnapshot(
            tuple(self._visited),
            tuple(self._frontier),
            tuple((url, self._pages[url]) for url in sorted(self._pages)),
        )

    def checkpoint(self) -> tuple[CrawlSnapshot, object]:
        """Capture crawl and private runtime state."""
        return self.snapshot(), self._runtime.checkpoint()


def _fixture_fetcher(url: str) -> CrawlPage:
    pages = {
        "https://example.invalid/": CrawlPage(
            b"root", ("/about", "https://example.invalid/about#fragment")
        ),
        "https://example.invalid/about": CrawlPage(b"about", ()),
    }
    return pages[url]


def create_web_crawl(
    seeds: Sequence[str] = ("https://example.invalid/",),
    fetcher: Fetcher = _fixture_fetcher,
    *,
    max_pages: int = 16,
    clock: Clock | None = None,
) -> WebCrawlEngine:
    """Create an offline, injected-fetcher crawl job."""
    return WebCrawlEngine(seeds, fetcher, max_pages=max_pages, clock=clock)
