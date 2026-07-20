"""Byte-oriented regular-expression scanning with explicit boundary halos."""

from __future__ import annotations

import re
from dataclasses import dataclass

from treemendous.applications._shared.claiming import ClaimUnavailableError, WorkClaim
from treemendous.applications._shared.clock import Clock
from treemendous.applications.partitioning._runtime import PartitionRuntime, positive


@dataclass(frozen=True, order=True)
class RegexMatch:
    """Absolute byte offsets and bytes for one deduplicated match."""

    start: int
    end: int
    value: bytes


@dataclass(frozen=True)
class RegexScanSnapshot:
    """Detached scan results and halo configuration."""

    halo: int
    matches: tuple[RegexMatch, ...]


class RegexScanEngine:
    """Scan claimed byte chunks while retaining cross-boundary matches.

    A match is owned by the chunk containing its start offset. The explicit
    halo must be large enough for the caller's pattern; Python regexes do not
    expose a reliable maximum width for every construct. State is process-local
    and distributed result stores must enforce claim fencing tokens.
    """

    def __init__(
        self,
        data: bytes,
        pattern: bytes,
        *,
        halo: int,
        flags: int = 0,
        clock: Clock | None = None,
    ) -> None:
        if not isinstance(data, bytes) or not data:
            raise ValueError("data must be nonempty bytes")
        if not isinstance(pattern, bytes) or not pattern:
            raise ValueError("pattern must be nonempty bytes")
        if type(halo) is not int or halo < 0:
            raise ValueError("halo must be a non-negative integer")
        if type(flags) is not int:
            raise TypeError("flags must be an integer")
        self._data = data
        try:
            self._pattern = re.compile(pattern, flags)
        except re.error as exc:
            raise ValueError(f"invalid byte pattern: {exc}") from exc
        if self._pattern.match(b"") is not None:
            raise ValueError("patterns that match empty bytes are not supported")
        self._halo = halo
        self._matches: dict[tuple[int, int, bytes], RegexMatch] = {}
        self._runtime = PartitionRuntime(len(data), clock=clock)

    def claim(self, owner: str, length: int) -> WorkClaim:
        """Claim the next core byte chunk."""
        return self._runtime.claim(owner, length)

    def scan_claim(self, claim: WorkClaim) -> tuple[RegexMatch, ...]:
        """Scan a core plus halos and merge only core-owned match starts."""
        scan_start = max(0, claim.span.start - self._halo)
        scan_end = min(len(self._data), claim.span.end + self._halo)
        local = self._data[scan_start:scan_end]
        found: list[RegexMatch] = []
        for match in self._pattern.finditer(local):
            start = scan_start + match.start()
            end = scan_start + match.end()
            if claim.span.start <= start < claim.span.end:
                item = RegexMatch(start, end, match.group())
                self._matches[(item.start, item.end, item.value)] = item
                found.append(item)
        self._runtime.complete(claim, "scanned", {"matches": len(found)})
        return tuple(sorted(found))

    def run(self, *, chunk_size: int = 4096, owner: str = "local") -> tuple[RegexMatch, ...]:
        """Scan every byte chunk and return globally ordered unique matches."""
        positive(chunk_size, "chunk_size")
        while True:
            try:
                claim = self.claim(owner, chunk_size)
            except ClaimUnavailableError:
                break
            self.scan_claim(claim)
        return tuple(sorted(self._matches.values()))

    def snapshot(self) -> RegexScanSnapshot:
        """Return immutable deduplicated scan state."""
        return RegexScanSnapshot(self._halo, tuple(sorted(self._matches.values())))

    def checkpoint(self) -> tuple[RegexScanSnapshot, object]:
        """Capture scan results with claim and event state."""
        return self.snapshot(), self._runtime.checkpoint()


def create_regex_scan(
    data: bytes = b"aa boundary match zz",
    pattern: bytes = b"boundary match",
    *,
    halo: int = 32,
    flags: int = 0,
    clock: Clock | None = None,
) -> RegexScanEngine:
    """Create a reusable byte regex scan job."""
    return RegexScanEngine(data, pattern, halo=halo, flags=flags, clock=clock)
