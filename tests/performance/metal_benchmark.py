#!/usr/bin/env python3
"""Correctness-checked experimental Metal benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from tests.performance.accelerator_benchmark import (
    HardwareUnavailableError,
    benchmark_accelerator,
    write_and_print,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--intervals", type=int, default=64)
    parser.add_argument("--operations", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = benchmark_accelerator(
            "metal_boundary_summary",
            samples=args.samples,
            warmups=args.warmups,
            intervals=args.intervals,
            operations=args.operations,
        )
    except HardwareUnavailableError as exc:
        print(f"Metal benchmark unavailable: {exc}")
        return 2
    write_and_print(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
