"""Validated benchmark for deterministic packet payload reassembly."""

from __future__ import annotations

from random import Random
from typing import Any

from tests.oracles.applications.catalogs.packet_sequence_reassembly import assemble
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from treemendous.applications.catalogs.packet_sequence_reassembly import (
    PacketReassemblyCatalog,
    PacketRecord,
    ReassemblyResult,
)

_MAX_OPERATIONS = 10_000


def _parameters(operations: int, seed: int) -> Random:
    if isinstance(operations, bool) or not isinstance(operations, int):
        raise TypeError("operations must be an integer")
    if not 1 <= operations <= _MAX_OPERATIONS:
        raise ValueError(f"operations must be between 1 and {_MAX_OPERATIONS}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return Random(seed)


def _identity(record: PacketRecord) -> tuple[Any, ...]:
    return (
        record.handle.owner,
        record.handle.sequence,
        "engine-lineage",
    )


def _record(record: PacketRecord) -> tuple[Any, ...]:
    fragment = record.payload
    return (
        *_identity(record),
        record.start,
        record.end,
        record.insertion_order,
        fragment.flow_id,
        fragment.payload,
    )


def _state(catalog: PacketReassemblyCatalog) -> dict[str, Any]:
    snapshot = catalog.snapshot()
    return {
        "records": tuple(_record(record) for record in snapshot.records),
        "next_sequences": snapshot.next_sequences,
        "next_insertion_order": snapshot.next_insertion_order,
    }


def _result(result: ReassemblyResult) -> dict[str, Any]:
    return {
        "flow_id": result.flow_id,
        "start": result.start,
        "end": result.end,
        "fragments": tuple(_record(record) for record in result.fragments),
        "gaps": tuple((gap.start, gap.end) for gap in result.gaps),
        "duplicate_coverage": tuple(
            (
                segment.start,
                segment.end,
                tuple(
                    sorted(
                        (handle.owner, handle.sequence, "engine-lineage")
                        for handle in segment.record_ids
                    )
                ),
            )
            for segment in result.duplicate_coverage
        ),
        "payload": result.payload,
        "complete": result.complete,
    }


def _record_overlaps(record: tuple[Any, ...], start: int, end: int) -> bool:
    left = record[3]
    right = record[4]
    if isinstance(left, bool) or not isinstance(left, int):
        raise TypeError("record start evidence must be an integer")
    if isinstance(right, bool) or not isinstance(right, int):
        raise TypeError("record end evidence must be an integer")
    return left < end and start < right


def _gaps(missing: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    if not missing:
        return ()
    result: list[tuple[int, int]] = []
    left = previous = missing[0]
    for point in missing[1:]:
        if point != previous + 1:
            result.append((left, previous + 1))
            left = point
        previous = point
    result.append((left, previous + 1))
    return tuple(result)


def run_benchmark(operations: int = 500, seed: int = 0) -> ApplicationSample:
    """Run bounded reassemblies and attest fragments, gaps, duplicates, and data."""
    random = _parameters(operations, seed)
    catalog = PacketReassemblyCatalog()
    fragments: list[tuple[int, bytes]] = []
    for index in range(200):
        sequence = index * 8
        payload = b"abcdefgh"
        catalog.add("flow", sequence, payload)
        fragments.append((sequence, payload))

    commands = tuple(random.randrange(191) * 8 for _ in range(operations))
    expected_state = _state(catalog)
    expected_records = expected_state["records"]

    def execute() -> tuple[ReassemblyResult, ...]:
        return tuple(catalog.assemble("flow", start, start + 80) for start in commands)

    def observe(raw: tuple[ReassemblyResult, ...]) -> ApplicationOutcome:
        results = tuple(_result(result) for result in raw)
        return ApplicationOutcome(
            results,
            _state(catalog),
            {
                "reassembly_calls": operations,
                "assembled_bytes": sum(
                    len(result["payload"] or b"") for result in results
                ),
                "returned_fragments": sum(
                    len(result["fragments"]) for result in results
                ),
            },
        )

    def oracle() -> ApplicationOutcome:
        results: list[dict[str, Any]] = []
        for start in commands:
            end = start + 80
            payload, missing = assemble(fragments, start, end)
            selected = tuple(
                record
                for record in expected_records
                if _record_overlaps(record, start, end)
            )
            results.append(
                {
                    "flow_id": "flow",
                    "start": start,
                    "end": end,
                    "fragments": selected,
                    "gaps": _gaps(missing),
                    "duplicate_coverage": (),
                    "payload": payload,
                    "complete": not missing,
                }
            )
        return ApplicationOutcome(
            tuple(results),
            expected_state,
            {
                "reassembly_calls": operations,
                "assembled_bytes": sum(
                    len(result["payload"] or b"") for result in results
                ),
                "returned_fragments": sum(
                    len(result["fragments"]) for result in results
                ),
            },
        )

    return run_application_case(
        scenario_id="packet-sequence-reassembly",
        operations=operations,
        execute=execute,
        observe=observe,
        oracle=oracle,
    )


def run_smoke(iterations: int = 500, seed: int = 0) -> ApplicationSample:
    """Delegate the legacy smoke entry point to the validated benchmark."""
    return run_benchmark(operations=iterations, seed=seed)


if __name__ == "__main__":
    print(run_smoke().to_dict())
