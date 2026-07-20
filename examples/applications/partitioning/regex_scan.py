#!/usr/bin/env python3
"""Run a boundary-aware byte regex scan from any working directory."""

from treemendous.applications.partitioning.regex_scan import RegexScanEngine


def main() -> None:
    matches = RegexScanEngine(b"xxneedlezz", b"needle", halo=6).run(chunk_size=4)
    if tuple((item.start, item.end) for item in matches) != ((2, 8),):
        raise RuntimeError("unexpected regex result")
    print("regex-scan: boundary match [2, 8)")


if __name__ == "__main__":
    main()
