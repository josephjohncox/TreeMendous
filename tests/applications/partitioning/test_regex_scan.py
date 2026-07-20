"""Regex-scan engine contracts."""

import pytest

from tests.oracles.applications.partitioning.regex_scan import expected_matches
from treemendous.applications.partitioning.regex_scan import RegexScanEngine


def test_halos_find_boundary_matches_once() -> None:
    data = b"xxABCDyyABCDz"
    engine = RegexScanEngine(data, b"ABCD", halo=4)
    observed = tuple(
        (item.start, item.end, item.value) for item in engine.run(chunk_size=3)
    )
    assert observed == expected_matches(data, b"ABCD")
    assert len(observed) == 2


def test_regex_scan_rejects_unsafe_patterns_and_halos() -> None:
    with pytest.raises(ValueError, match="empty bytes"):
        RegexScanEngine(b"abc", b"a*", halo=1)
    with pytest.raises(ValueError, match="halo"):
        RegexScanEngine(b"abc", b"a", halo=-1)
