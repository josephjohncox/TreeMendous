"""Deterministic token-index document search partitioned by document ordinal."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from treemendous.applications._shared.claiming import ClaimUnavailableError, WorkClaim
from treemendous.applications._shared.clock import Clock
from treemendous.applications.partitioning._runtime import PartitionRuntime, positive

_TOKEN = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class SearchHit:
    """One document matching every normalized query token."""

    document_id: int
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class DocumentSearchSnapshot:
    """Detached search-job state in deterministic document order."""

    query: tuple[str, ...]
    hits: tuple[SearchHit, ...]
    claimed_documents: int


class DocumentSearchEngine:
    """Build and query an immutable token index using claimed document bands.

    Coordination and results are process-local. A distributed adapter must
    persist checkpoints and fence commits using ``WorkClaim.fencing_token``.
    """

    def __init__(
        self,
        documents: Mapping[int, str],
        query: str | Sequence[str],
        *,
        clock: Clock | None = None,
    ) -> None:
        if not isinstance(documents, Mapping):
            raise TypeError("documents must be a mapping")
        if not documents:
            raise ValueError("documents must not be empty")
        normalized: list[tuple[int, str]] = []
        for document_id, text in documents.items():
            if type(document_id) is not int or document_id < 0:
                raise ValueError("document IDs must be non-negative integers")
            if not isinstance(text, str):
                raise TypeError("document text must be a string")
            normalized.append((document_id, text))
        self._documents = tuple(sorted(normalized))
        raw_query = query.split() if isinstance(query, str) else query
        if isinstance(raw_query, (str, bytes)) or not isinstance(raw_query, Sequence):
            raise TypeError("query must be a string or sequence of strings")
        tokens: list[str] = []
        for item in raw_query:
            if not isinstance(item, str):
                raise TypeError("query tokens must be strings")
            tokens.extend(token.lower() for token in _TOKEN.findall(item))
        self._query = tuple(dict.fromkeys(tokens))
        if not self._query:
            raise ValueError("query must contain at least one token")

        postings: dict[str, list[int]] = {}
        tokenized: dict[int, tuple[str, ...]] = {}
        for document_id, text in self._documents:
            document_tokens = tuple(token.lower() for token in _TOKEN.findall(text))
            tokenized[document_id] = document_tokens
            for token in sorted(set(document_tokens)):
                postings.setdefault(token, []).append(document_id)
        self._postings = {key: tuple(value) for key, value in sorted(postings.items())}
        self._tokenized = tokenized
        self._matches: dict[int, SearchHit] = {}
        self._runtime = PartitionRuntime(len(self._documents), clock=clock)

    @property
    def index(self) -> Mapping[str, tuple[int, ...]]:
        """Return a detached deterministic token index."""
        return self._postings.copy()

    def claim(self, owner: str, length: int) -> WorkClaim:
        """Claim the next document-ordinal band."""
        return self._runtime.claim(owner, length)

    def search_claim(self, claim: WorkClaim) -> tuple[SearchHit, ...]:
        """Query one claimed band and merge its results idempotently."""

        def prepare() -> tuple[tuple[SearchHit, ...], dict[int, SearchHit]]:
            candidate_ids = set(self._postings.get(self._query[0], ()))
            for token in self._query[1:]:
                candidate_ids.intersection_update(self._postings.get(token, ()))
            hits = tuple(
                SearchHit(document_id, self._tokenized[document_id])
                for document_id, _ in self._documents[claim.span.start : claim.span.end]
                if document_id in candidate_ids
            )
            matches = self._matches.copy()
            matches.update((hit.document_id, hit) for hit in hits)
            return hits, matches

        prepared = self._runtime.execute_claim(
            claim,
            kind="searched",
            prepare=prepare,
            commit=lambda value: setattr(self, "_matches", value[1]),
            result=lambda value: {"matches": len(value[0])},
        )
        return prepared[0]

    def run(
        self, *, shard_size: int = 64, owner: str = "local"
    ) -> tuple[SearchHit, ...]:
        """Execute all unclaimed bands and return merged ordered hits."""
        positive(shard_size, "shard_size")
        while True:
            try:
                claim = self.claim(owner, shard_size)
            except ClaimUnavailableError:
                break
            self.search_claim(claim)
        return tuple(self._matches[key] for key in sorted(self._matches))

    def _snapshot(self) -> DocumentSearchSnapshot:
        ledger = self._runtime.ledger.snapshot()
        claimed = len(self._documents) - ledger.diagnostics.available_work
        return DocumentSearchSnapshot(
            self._query,
            tuple(self._matches[key] for key in sorted(self._matches)),
            claimed,
        )

    def snapshot(self) -> DocumentSearchSnapshot:
        """Return detached query and result state."""
        return self._runtime.observe(self._snapshot)

    def audit_snapshot(self) -> tuple[DocumentSearchSnapshot, object]:
        """Capture non-restorable application and runtime audit evidence."""
        return self._runtime.audit_snapshot(self._snapshot)


def create_document_search(
    documents: Mapping[int, str] | None = None,
    query: str | Sequence[str] = "range",
    *,
    clock: Clock | None = None,
) -> DocumentSearchEngine:
    """Create a reusable document-search job."""
    selected = (
        {0: "range search", 1: "other document"} if documents is None else documents
    )
    return DocumentSearchEngine(selected, query, clock=clock)
