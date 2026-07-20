#!/usr/bin/env python3
"""Generate the documented application status and navigation catalogs."""

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
APPLICATION_DOCUMENT = ROOT / "docs/applications.md"
START = "<!-- BEGIN GENERATED SCENARIO STATUS -->"
END = "<!-- END GENERATED SCENARIO STATUS -->"
APPLICATION_START = "<!-- BEGIN GENERATED APPLICATION INDEX -->"
APPLICATION_END = "<!-- END GENERATED APPLICATION INDEX -->"
INSERT_BEFORE = (
    "The benchmark suite executes the following 50 application-shaped scenarios"
)
APPLICATION_INSERT_BEFORE = "## Guarantees and limits"

FAMILY_SECTIONS = (
    (
        "partition",
        "Partitioning and work claiming",
        "treemendous.applications.partitioning",
    ),
    (
        "scheduling",
        "Scheduling and reservation",
        "treemendous.applications.scheduling",
    ),
    (
        "catalog",
        "Identity-preserving overlap catalogs",
        "treemendous.applications.catalogs",
    ),
    (
        "allocator",
        "Allocation and capacity tracking",
        "treemendous.applications.allocation",
    ),
    (
        "lease",
        "Numeric resource leasing",
        "treemendous.applications.leasing",
    ),
)


def _relative_target(reference: str) -> str:
    if reference.startswith("docs/"):
        return reference.removeprefix("docs/")
    return f"../{reference}"


def _engine_source(reference: str) -> str:
    module_name, separator, _ = reference.partition(":")
    if not separator:
        raise ValueError(f"invalid engine reference: {reference}")
    return f"{module_name.replace('.', '/')}.py"


def _evidence_link(reference: str | None, field_name: str) -> str:
    if reference is None:
        return "missing"
    if field_name == "engine":
        label = reference.rpartition(":")[2]
        target = _engine_source(reference)
    else:
        label = field_name
        target = reference
    return f"[{label}]({_relative_target(target)})"


def render_status_block() -> str:
    """Return the complete deterministic generated Markdown status block."""
    validate_catalog_evidence(root=ROOT)
    completed = sum(spec.status is ScenarioStatus.COMPLETE for spec in SCENARIO_SPECS)
    lines = [
        START,
        "## Application implementation status",
        "",
        f"Current completion: **{completed}/{len(SCENARIO_SPECS)}** real engines.",
        "A legacy backend trace is not implementation evidence. An entry becomes",
        "`COMPLETE` only when its engine, example, independent oracle, benchmark,",
        "and scenario documentation are all registered and resolve.",
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
                    _evidence_link(spec.engine, "engine"),
                    _evidence_link(spec.example, "example"),
                    _evidence_link(spec.oracle, "oracle"),
                    _evidence_link(spec.benchmark, "benchmark"),
                    _evidence_link(spec.docs, "docs"),
                )
            )
            + " |"
        )
    lines.extend((END, ""))
    return "\n".join(lines)


def render_application_index() -> str:
    """Return family-grouped navigation generated from the canonical specs."""
    validate_catalog_evidence(root=ROOT)
    lines = [APPLICATION_START]
    for family, heading, package in FAMILY_SECTIONS:
        specs = tuple(spec for spec in SCENARIO_SPECS if spec.family == family)
        lines.extend(
            (
                f"## {heading}",
                "",
                f"Package: `{package}`. Registered engines: {len(specs)}.",
                "",
                "| Application | Purpose | Example |",
                "| --- | --- | --- |",
            )
        )
        for spec in specs:
            title = _evidence_link(spec.docs, spec.title)
            example = _evidence_link(spec.example, "example")
            lines.append(f"| {title} | {spec.description} | {example} |")
        lines.append("")
    lines.extend((APPLICATION_END, ""))
    return "\n".join(lines)


def _updated_block(
    current: str,
    *,
    block: str,
    start: str,
    end: str,
    insert_before: str,
    description: str,
) -> str:
    start_count = current.count(start)
    end_count = current.count(end)
    if start_count != end_count:
        raise ValueError(f"{description} markers are incomplete")
    if start_count > 1:
        raise ValueError(f"{description} contains duplicate generated blocks")
    if start_count:
        prefix, remainder = current.split(start, 1)
        _, suffix = remainder.split(end, 1)
        return prefix + block.rstrip("\n") + suffix
    if insert_before not in current:
        raise ValueError(f"{description} insertion anchor is missing")
    prefix, suffix = current.split(insert_before, 1)
    return prefix + block + "\n" + insert_before + suffix


def updated_document(current: str) -> str:
    """Return the use-case document with exactly one current status block."""
    return _updated_block(
        current,
        block=render_status_block(),
        start=START,
        end=END,
        insert_before=INSERT_BEFORE,
        description="scenario status document",
    )


def updated_application_document(current: str) -> str:
    """Return the application index with exactly one current generated block."""
    return _updated_block(
        current,
        block=render_application_index(),
        start=APPLICATION_START,
        end=APPLICATION_END,
        insert_before=APPLICATION_INSERT_BEFORE,
        description="application index document",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail without writing when either generated catalog has drifted",
    )
    args = parser.parse_args(argv)

    current = DOCUMENT.read_text(encoding="utf-8")
    application_current = APPLICATION_DOCUMENT.read_text(encoding="utf-8")
    try:
        expected = updated_document(current)
        application_expected = updated_application_document(application_current)
    except ValueError as exc:
        print(f"scenario catalog validation failed: {exc}", file=sys.stderr)
        return 2

    stale = current != expected
    application_stale = application_current != application_expected
    if args.check:
        if stale or application_stale:
            documents = []
            if stale:
                documents.append("docs/use-cases.md")
            if application_stale:
                documents.append("docs/applications.md")
            print(
                f"{', '.join(documents)} generated catalog is stale; run "
                "python scripts/generate_scenario_catalog.py",
                file=sys.stderr,
            )
            return 1
        return 0

    if stale:
        DOCUMENT.write_text(expected, encoding="utf-8")
    if application_stale:
        APPLICATION_DOCUMENT.write_text(application_expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
