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
    assert (
        first.result_checksum,
        first.state_checksum,
        first.counters_checksum,
        first.evidence_checksum,
    ) == (
        second.result_checksum,
        second.state_checksum,
        second.counters_checksum,
        second.evidence_checksum,
    )


@pytest.mark.parametrize("benchmark", CATALOG_BENCHMARKS)
def test_catalog_benchmark_has_uniform_bounded_api(benchmark: ModuleType) -> None:
    parameters = inspect.signature(benchmark.run_benchmark).parameters
    assert tuple(parameters) == ("operations", "seed")
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
    assert received == [(3, 11)]
