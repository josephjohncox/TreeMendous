"""Collected contracts for all allocation application benchmarks."""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType

import pytest

from tests.performance.applications.harness import ApplicationSample
from treemendous.applications import SCENARIO_SPECS

MODULES = {
    "cdn-byte-range-cache": "cdn_byte_range_cache",
    "database-page-allocation": "database_page_allocation",
    "disk-block-allocation": "disk_block_allocation",
    "gpu-memory-arena": "gpu_memory_arena",
    "heap-free-space": "heap_free_space",
    "object-store-multipart-ranges": "object_store_multipart_ranges",
    "ring-buffer-sequences": "ring_buffer_sequences",
    "virtual-address-space": "virtual_address_space",
}


def _module(name: str) -> ModuleType:
    return importlib.import_module(f"tests.performance.applications.allocation.{name}")


def test_contract_covers_every_registered_allocation_benchmark() -> None:
    registered = {
        spec.id for spec in SCENARIO_SPECS if spec.category == "allocation_churn"
    }
    assert set(MODULES) == registered


@pytest.mark.parametrize(("scenario_id", "module_name"), MODULES.items())
def test_uniform_entry_point_returns_reproducible_attestation(
    scenario_id: str, module_name: str
) -> None:
    module = _module(module_name)
    signature = inspect.signature(module.run_benchmark)
    assert list(signature.parameters) == ["operations", "seed"]
    assert signature.parameters["operations"].default == module.DEFAULT_OPERATIONS
    assert 0 < module.DEFAULT_OPERATIONS <= 1_000
    assert signature.parameters["seed"].default == 42

    first = module.run_benchmark(operations=3, seed=91)
    second = module.run_benchmark(operations=3, seed=91)

    assert isinstance(first, ApplicationSample)
    assert first.scenario_id == scenario_id
    assert first.operations == 3
    assert first.validated
    assert first.result_checksum == second.result_checksum
    assert first.state_checksum == second.state_checksum
    assert first.counters_checksum == second.counters_checksum
    assert first.evidence_checksum == second.evidence_checksum


@pytest.mark.parametrize("module_name", MODULES.values())
def test_seconds_only_smoke_entry_point_delegates(module_name: str) -> None:
    sample = _module(module_name).run_smoke(operations=2)
    assert isinstance(sample, ApplicationSample)
    assert sample.operations == 2
    assert sample.validated


@pytest.mark.parametrize("module_name", MODULES.values())
def test_entry_point_rejects_nonpositive_operations(module_name: str) -> None:
    with pytest.raises(ValueError, match="operations must be positive"):
        _module(module_name).run_benchmark(operations=0)
