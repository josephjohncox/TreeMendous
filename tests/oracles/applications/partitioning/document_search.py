"""Independent exhaustive oracle for document search."""

from __future__ import annotations

import re
from collections.abc import Mapping

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize text using the scenario's documented ASCII-token rules."""
    return tuple(token.lower() for token in _TOKEN.findall(text))


def expected_hits(documents: Mapping[int, str], query: str) -> tuple[int, ...]:
    """Return ordered document IDs containing every normalized query token."""
    tokens = tuple(dict.fromkeys(tokenize(query)))
    return tuple(
        key
        for key, text in sorted(documents.items())
        if set(tokens) <= set(tokenize(text))
    )


def expected_index(
    documents: Mapping[int, str],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Build the complete deterministic inverted index by exhaustive scanning."""
    postings: dict[str, list[int]] = {}
    for document_id, text in sorted(documents.items()):
        for token in sorted(set(tokenize(text))):
            postings.setdefault(token, []).append(document_id)
    return tuple((token, tuple(ids)) for token, ids in sorted(postings.items()))
