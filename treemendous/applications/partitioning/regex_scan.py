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
    """Scan claimed byte chunks with full-buffer regular-expression semantics.

    A match is owned by the chunk containing its start offset. Evaluation uses
    the original buffer so chunk edges never become artificial anchors or
    lookaround boundaries. ``halo`` is retained as audit configuration. State
    is process-local and distributed result stores must enforce claim fencing.
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
        if not isinstance(flags, int) or isinstance(flags, bool):
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
        """Scan with full-buffer context and merge only core-owned starts."""
        def prepare() -> tuple[
            tuple[RegexMatch, ...], dict[tuple[int, int, bytes], RegexMatch]
        ]:
            found: list[RegexMatch] = []
            matches = self._matches.copy()
            for match in self._pattern.finditer(self._data):
                start = match.start()
                if claim.span.start <= start < claim.span.end:
                    item = RegexMatch(start, match.end(), match.group())
                    matches[(item.start, item.end, item.value)] = item
                    found.append(item)
            return tuple(sorted(found)), matches

        prepared = self._runtime.execute_claim(
            claim,
            kind="scanned",
            prepare=prepare,
            commit=lambda value: setattr(self, "_matches", value[1]),
            result=lambda value: {"matches": len(value[0])},
        )
        return prepared[0]

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

    def _snapshot(self) -> RegexScanSnapshot:
        return RegexScanSnapshot(self._halo, tuple(sorted(self._matches.values())))

    def snapshot(self) -> RegexScanSnapshot:
        """Return immutable deduplicated scan state."""
        return self._runtime.observe(self._snapshot)

    def audit_snapshot(self) -> tuple[RegexScanSnapshot, object]:
        """Capture non-restorable application and runtime audit evidence."""
        return self._runtime.audit_snapshot(self._snapshot)


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
