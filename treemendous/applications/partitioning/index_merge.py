"""Term-band search-index merge with posting deduplication and order checks."""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from treemendous.applications._shared.claiming import ClaimUnavailableError, WorkClaim
from treemendous.applications._shared.clock import Clock
from treemendous.applications.partitioning._runtime import (
    PartitionRuntime,
    nonempty,
    positive,
)


@dataclass(frozen=True)
class TermPostings:
    """One term and its strictly ordered deduplicated postings."""

    term: str
    postings: tuple[int, ...]


@dataclass(frozen=True)
class IndexMergeSnapshot:
    """Immutable terms and merged postings accumulated so far."""

    terms: tuple[str, ...]
    merged: tuple[TermPostings, ...]


class IndexMergeEngine:
    """Merge sorted posting lists independently across claimed term bands.

    Source segments are copied and validated before work begins. Coordination
    and merged state are process-local; distributed stores must use claim
    fencing tokens and atomically publish complete term bands.
    """

    def __init__(
        self,
        segments: object,
        *,
        clock: Clock | None = None,
    ) -> None:
        if isinstance(segments, (str, bytes)) or not isinstance(segments, Sequence):
            raise TypeError("segments must be a sequence")
        if not segments:
            raise ValueError("segments must not be empty")
        checked: list[dict[str, tuple[int, ...]]] = []
        for raw_segment in cast(Sequence[object], segments):
            if not isinstance(raw_segment, Mapping):
                raise TypeError("each segment must be a mapping")
            segment = cast(Mapping[object, object], raw_segment)
            copied: dict[str, tuple[int, ...]] = {}
            for raw_term, raw_postings in segment.items():
                if not isinstance(raw_term, str):
                    raise TypeError("terms must be strings")
                term = nonempty(raw_term, "term")
                if isinstance(raw_postings, (str, bytes)) or not isinstance(
                    raw_postings, Sequence
                ):
                    raise TypeError("postings must be a sequence")
                postings_list: list[int] = []
                for value in cast(Sequence[object], raw_postings):
                    if type(value) is not int or value < 0:
                        raise ValueError("postings must be non-negative integers")
                    postings_list.append(value)
                postings = tuple(postings_list)
                if any(left > right for left, right in zip(postings, postings[1:])):
                    raise ValueError("source posting lists must be sorted")
                copied[term] = postings
            checked.append(copied)
        terms = sorted({term for segment in checked for term in segment})
        if not terms:
            raise ValueError("segments must contain at least one term")
        self._segments = tuple(checked)
        self._terms = tuple(terms)
        self._merged: dict[str, tuple[int, ...]] = {}
        self._runtime = PartitionRuntime(len(terms), clock=clock)

    def claim(self, owner: str, length: int) -> WorkClaim:
        """Claim lexicographically contiguous term ordinals."""
        return self._runtime.claim(owner, length)

    def merge_claim(self, claim: WorkClaim) -> tuple[TermPostings, ...]:
        """Merge and validate all terms in one claimed band."""
        staged: list[TermPostings] = []
        for term in self._terms[claim.span.start : claim.span.end]:
            streams = [segment[term] for segment in self._segments if term in segment]
            merged = tuple(dict.fromkeys(heapq.merge(*streams)))
            if any(left >= right for left, right in zip(merged, merged[1:])):
                self._runtime.abandon(claim)
                raise RuntimeError("merged posting list is not strictly ordered")
            staged.append(TermPostings(term, merged))
        for item in staged:
            self._merged[item.term] = item.postings
        self._runtime.complete(claim, "merged", {"terms": len(staged)})
        return tuple(staged)

    def run(self, *, band_size: int = 64) -> tuple[TermPostings, ...]:
        """Merge every remaining term band in lexical order."""
        positive(band_size, "band_size")
        while True:
            try:
                claim = self.claim("local", band_size)
            except ClaimUnavailableError:
                break
            self.merge_claim(claim)
        return tuple(TermPostings(term, self._merged[term]) for term in sorted(self._merged))

    def snapshot(self) -> IndexMergeSnapshot:
        """Return immutable source-term and merged state."""
        return IndexMergeSnapshot(
            self._terms,
            tuple(TermPostings(term, self._merged[term]) for term in sorted(self._merged)),
        )

    def checkpoint(self) -> tuple[IndexMergeSnapshot, object]:
        """Capture merged postings and private runtime state."""
        return self.snapshot(), self._runtime.checkpoint()


def create_index_merge(
    segments: Sequence[Mapping[str, Sequence[int]]] = (
        {"range": (1, 3), "tree": (2,)},
        {"range": (2, 3, 4), "set": (5,)},
    ),
    *,
    clock: Clock | None = None,
) -> IndexMergeEngine:
    """Create a validated posting-list merge job."""
    return IndexMergeEngine(segments, clock=clock)
