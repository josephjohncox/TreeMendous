#!/usr/bin/env python3
"""Generate the documented application implementation-status catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from treemendous.applications import (
    SCENARIO_SPECS,
    ScenarioStatus,
    validate_catalog_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs/use-cases.md"
START = "<!-- BEGIN GENERATED SCENARIO STATUS -->"
END = "<!-- END GENERATED SCENARIO STATUS -->"
INSERT_BEFORE = (
    "The benchmark suite executes the following 50 application-shaped scenarios"
)


def _artifact_state(reference: str | None, field_name: str) -> str:
    if reference is None:
        return "missing"
    if field_name == "engine":
        return "present"
    return "present" if (ROOT / reference).is_file() else "missing"


def render_status_block() -> str:
    """Return the complete deterministic generated Markdown block."""
    validate_catalog_evidence(root=ROOT)
    completed = sum(spec.status is ScenarioStatus.COMPLETE for spec in SCENARIO_SPECS)
    lines = [
        START,
        "## Application implementation status",
        "",
        f"Current completion: **{completed}/{len(SCENARIO_SPECS)}** real engines.",
        "A benchmark trace is not implementation evidence. An entry becomes",
        "`COMPLETE` only when its engine, example, independent oracle, benchmark,",
        "and scenario documentation are all registered.",
        "",
        "| Scenario | Family | Category | Status | Engine | Example | Oracle | Benchmark | Docs |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for spec in SCENARIO_SPECS:
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{spec.id}`",
                    f"`{spec.family}`",
                    f"`{spec.category}`",
                    f"`{spec.status.name}`",
                    _artifact_state(spec.engine, "engine"),
                    _artifact_state(spec.example, "example"),
                    _artifact_state(spec.oracle, "oracle"),
                    _artifact_state(spec.benchmark, "benchmark"),
                    _artifact_state(spec.docs, "docs"),
                )
            )
            + " |"
        )
    lines.extend((END, ""))
    return "\n".join(lines)


def updated_document(current: str) -> str:
    """Return document text with exactly one current generated block."""
    block = render_status_block()
    start_count = current.count(START)
    end_count = current.count(END)
    if start_count != end_count:
        raise ValueError("scenario status document markers are incomplete")
    if start_count > 1:
        raise ValueError("scenario status document contains duplicate generated blocks")
    if start_count:
        prefix, remainder = current.split(START, 1)
        _, suffix = remainder.split(END, 1)
        return prefix + block.rstrip("\n") + suffix
    if INSERT_BEFORE not in current:
        raise ValueError("scenario status insertion anchor is missing")
    prefix, suffix = current.split(INSERT_BEFORE, 1)
    return prefix + block + "\n" + INSERT_BEFORE + suffix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail without writing when the generated status block has drifted",
    )
    args = parser.parse_args(argv)

    current = DOCUMENT.read_text(encoding="utf-8")
    try:
        expected = updated_document(current)
    except ValueError as exc:
        print(f"scenario catalog validation failed: {exc}", file=sys.stderr)
        return 2
    if args.check:
        if current != expected:
            print(
                "docs/use-cases.md scenario status is stale; run "
                "python scripts/generate_scenario_catalog.py",
                file=sys.stderr,
            )
            return 1
        return 0
    if current != expected:
        DOCUMENT.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
