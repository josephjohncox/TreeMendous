"""Correctness-checked smoke workload for byte regex scanning."""

from tests.oracles.applications.partitioning.regex_scan import expected_matches
from treemendous.applications.partitioning.regex_scan import RegexScanEngine


def run_smoke() -> int:
    data = (b"0123456789needle" * 200) + b"tail"
    engine = RegexScanEngine(data, b"needle", halo=6)
    observed = tuple((m.start, m.end, m.value) for m in engine.run(chunk_size=17))
    if observed != expected_matches(data, b"needle"):
        raise AssertionError("regex smoke differs from whole-buffer oracle")
    return len(observed)
