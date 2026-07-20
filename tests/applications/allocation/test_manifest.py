"""Allocation-family implementation manifest contracts."""

from pathlib import Path

from treemendous.applications import (
    ScenarioStatus,
    list_scenarios,
    validate_catalog_evidence,
)
from treemendous.applications.allocation.manifest import EVIDENCE

EXPECTED_IDS = {
    "heap-free-space",
    "disk-block-allocation",
    "virtual-address-space",
    "database-page-allocation",
    "object-store-multipart-ranges",
    "cdn-byte-range-cache",
    "gpu-memory-arena",
    "ring-buffer-sequences",
}


def test_manifest_covers_exactly_allocator_ids_with_real_evidence() -> None:
    assert set(EVIDENCE) == EXPECTED_IDS
    specs = list_scenarios(family="allocator")
    assert {spec.id for spec in specs} == EXPECTED_IDS
    assert all(spec.status is ScenarioStatus.COMPLETE for spec in specs)
    validate_catalog_evidence(specs, root=Path(__file__).parents[3])
