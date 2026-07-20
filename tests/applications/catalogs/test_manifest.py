"""Catalog family evidence manifest contract."""

from pathlib import Path

from treemendous.applications import (
    ScenarioStatus,
    create_application,
    list_scenarios,
    validate_catalog_evidence,
)
from treemendous.applications.catalogs.manifest import EVIDENCE

CATALOG_IDS = {
    "genomic-annotation-overlap",
    "source-diagnostic-ranges",
    "filesystem-byte-locks",
    "database-key-range-locks",
    "packet-sequence-reassembly",
    "subtitle-cue-ranges",
    "video-edit-regions",
    "timeseries-alert-windows",
    "distributed-trace-spans",
    "morton-geospatial-ranges",
}


def test_manifest_exactly_completes_ten_catalog_scenarios() -> None:
    assert set(EVIDENCE) == CATALOG_IDS
    specs = list_scenarios(status=ScenarioStatus.COMPLETE, family="catalog")
    assert {spec.id for spec in specs} == CATALOG_IDS
    validate_catalog_evidence(specs, root=Path(__file__).parents[3])
    for scenario_id in CATALOG_IDS:
        assert create_application(scenario_id) is not None
