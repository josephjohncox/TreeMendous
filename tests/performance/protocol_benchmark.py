#!/usr/bin/env python3
"""CLI for oracle-validated, locally directional CPU benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from treemendous import list_available_backends

from tests.performance.harness import benchmark_backends
from tests.performance.workload import fragmented_workload, scheduling_workload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operations", type=int, default=500)
    parser.add_argument("--intervals", type=int, default=64)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--backends", nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="retained for compatibility; validation is always mandatory",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="small correctness-checked CPU smoke run (still uses 20 samples)",
    )
    parser.add_argument(
        "--scheduling",
        action="store_true",
        help="run 1/8/64-core constrained scheduling workloads",
    )
    return parser


def _default_backends() -> tuple[str, ...]:
    available = list_available_backends()
    preferred = (
        "py_boundary",
        "py_avl_earliest",
        "py_summary",
        "py_treap",
        "py_boundary_summary",
        "cpp_boundary",
        "cpp_boundary_optimized",
    )
    return tuple(backend for backend in preferred if backend in available)


def _print_report(report: dict[str, Any]) -> None:
    dataset = report["dataset"]
    print(f"\n{report['workload']}: {report['label']}")
    print(
        "  dataset: "
        f"{dataset['actual_interval_count']} actual intervals, "
        f"coordinate extent {dataset['coordinate_extent']}, "
        f"{dataset['timed_operations']} operations"
    )
    for backend, result in report["results"].items():
        timing = result["execution"]
        low, high = timing["confidence_95_ns"]
        validation = result["validation"]
        print(
            f"  {backend}: median={timing['median_ns'] / 1e6:.3f} ms "
            f"(95% median CI {low / 1e6:.3f}..{high / 1e6:.3f} ms, "
            f"MAD={timing['median_absolute_deviation_ns'] / 1e6:.3f} ms)"
        )
        print(
            "    validated: "
            f"success={validation['successful_operations']}, "
            f"no-op={validation['no_op_operations']}, "
            f"errors={validation['error_operations']}, "
            f"final_intervals={validation['actual_interval_count']}, "
            f"total={validation['total_available']}, "
            f"state={validation['state_checksum'][:12]}, "
            f"queries={validation['query_checksum'][:12]}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    backend_ids = tuple(args.backends or _default_backends())
    if args.quick:
        backend_ids = ("py_boundary",)
        args.operations = min(args.operations, 80)
        args.intervals = min(args.intervals, 16)
        args.warmups = 1
        args.processes = 1
    if not backend_ids:
        raise SystemExit("no semantically validated CPU backends are available")

    workloads = [
        fragmented_workload(
            interval_count=args.intervals,
            operation_count=args.operations,
        )
    ]
    if args.scheduling:
        workloads.extend(
            scheduling_workload(
                cores=cores,
                occupancy=occupancy,
                jobs=args.operations,
                seed=42 + cores + int(occupancy * 100),
            )
            for cores in (1, 8, 64)
            for occupancy in (0.25, 0.50, 0.75)
        )

    reports = [
        benchmark_backends(
            backend_ids,
            workload,
            samples=args.samples,
            warmups=args.warmups,
            processes=args.processes,
        )
        for workload in workloads
    ]
    for report in reports:
        _print_report(report)

    output = {
        "schema": "treemendous-correctness-checked-benchmark-v1",
        "reports": reports,
    }
    if args.output:
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
