"""Focused contracts for correctness-attested catalog benchmarks."""

from __future__ import annotations

import inspect
from types import ModuleType

import pytest

from tests.performance.applications.catalogs import (
    database_key_range_locks,
    distributed_trace_spans,
    filesystem_byte_locks,
    genomic_annotation_overlap,
    morton_geospatial_ranges,
    packet_sequence_reassembly,
    source_diagnostic_ranges,
    subtitle_cue_ranges,
    timeseries_alert_windows,
    video_edit_regions,
)
from tests.performance.applications.harness import ApplicationSample
from treemendous.applications.catalogs.morton_geospatial_ranges import (
    MortonGeospatialCatalog,
)

CATALOG_BENCHMARKS = (
    database_key_range_locks,
    distributed_trace_spans,
    filesystem_byte_locks,
    genomic_annotation_overlap,
    morton_geospatial_ranges,
    packet_sequence_reassembly,
    source_diagnostic_ranges,
    subtitle_cue_ranges,
    timeseries_alert_windows,
    video_edit_regions,
)


@pytest.mark.parametrize("benchmark", CATALOG_BENCHMARKS)
def test_catalog_benchmark_returns_repeatable_validated_evidence(
    benchmark: ModuleType,
) -> None:
    first = benchmark.run_benchmark(operations=7, seed=19)
    second = benchmark.run_benchmark(operations=7, seed=19)

    assert isinstance(first, ApplicationSample)
    assert first.validated
    assert first.operations == 7
    assert first.execution_ns >= 0
    first_checksums = (
        first.result_checksum,
        first.state_checksum,
        first.counters_checksum,
        first.evidence_checksum,
    )
    second_checksums = (
        second.result_checksum,
        second.state_checksum,
        second.counters_checksum,
        second.evidence_checksum,
    )
    assert first_checksums == second_checksums


@pytest.mark.parametrize("benchmark", CATALOG_BENCHMARKS)
def test_catalog_benchmark_has_uniform_bounded_api(benchmark: ModuleType) -> None:
    parameters = inspect.signature(benchmark.run_benchmark).parameters
    parameter_names = tuple(parameters)
    expected_names = ("operations", "seed")
    assert parameter_names == expected_names
    assert parameters["seed"].default == 0

    with pytest.raises(ValueError, match="between"):
        benchmark.run_benchmark(operations=0, seed=0)
    with pytest.raises(ValueError, match="between"):
        benchmark.run_benchmark(operations=10_001, seed=0)
    with pytest.raises(TypeError, match="operations"):
        benchmark.run_benchmark(operations=True, seed=0)
    with pytest.raises(TypeError, match="seed"):
        benchmark.run_benchmark(operations=1, seed=True)


@pytest.mark.parametrize("benchmark", CATALOG_BENCHMARKS)
def test_legacy_smoke_entry_point_delegates_to_validated_benchmark(
    benchmark: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = object()
    received: list[tuple[int, int]] = []

    def fake_run_benchmark(*, operations: int, seed: int) -> object:
        received.append((operations, seed))
        return expected

    monkeypatch.setattr(benchmark, "run_benchmark", fake_run_benchmark)

    assert benchmark.run_smoke(iterations=3, seed=11) is expected
    expected_calls = [(3, 11)]
    assert received == expected_calls


def test_catalog_benchmark_rejects_corrupted_retained_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_add = MortonGeospatialCatalog.add

    def corrupted_add(
        self: MortonGeospatialCatalog,
        item_id: str,
        min_x: int,
        min_y: int,
        max_x: int,
        max_y: int,
        *,
        label: str,
    ) -> object:
        return original_add(
            self,
            item_id,
            min_x,
            min_y,
            max_x,
            max_y,
            label=f"CORRUPTED:{label}",
        )

    monkeypatch.setattr(MortonGeospatialCatalog, "add", corrupted_add)

    with pytest.raises(AssertionError, match="evidence differs"):
        morton_geospatial_ranges.run_benchmark(operations=3, seed=5)


def test_catalog_benchmark_rejects_foreign_instance_query_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foreign = MortonGeospatialCatalog(bits=10)
    for index in range(200):
        x = index % 50 * 10
        y = index // 50 * 20
        foreign.add(f"g{index}", x, y, x + 8, y + 8, label="region")
    foreign_search = foreign.search

    def search_foreign(
        _self: MortonGeospatialCatalog,
        min_x: int,
        min_y: int,
        max_x: int,
        max_y: int,
    ) -> object:
        return foreign_search(min_x, min_y, max_x, max_y)

    monkeypatch.setattr(MortonGeospatialCatalog, "search", search_foreign)

    with pytest.raises(AssertionError, match="evidence differs"):
        morton_geospatial_ranges.run_benchmark(operations=3, seed=5)


def test_catalog_record_serialization_distinguishes_two_lineages() -> None:
    left_catalog = MortonGeospatialCatalog(bits=10)
    right_catalog = MortonGeospatialCatalog(bits=10)
    left_catalog.add("region", 1, 2, 3, 4, label="same")
    right_catalog.add("region", 1, 2, 3, 4, label="same")
    left = left_catalog.snapshot().records[0]
    right = right_catalog.snapshot().records[0]

    left_evidence = morton_geospatial_ranges._record(left)
    right_evidence = morton_geospatial_ranges._record(right)

    assert left.handle.lineage != right.handle.lineage
    assert left_evidence[2] == str(left.handle.lineage)
    assert right_evidence[2] == str(right.handle.lineage)
    assert left_evidence != right_evidence
