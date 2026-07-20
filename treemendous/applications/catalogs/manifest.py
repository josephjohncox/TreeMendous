"""Implementation evidence for exactly the ten overlap/catalog scenarios."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


def _evidence(module: str, factory: str, scenario_id: str) -> Mapping[str, str]:
    artifact_name = scenario_id.replace("-", "_")
    return MappingProxyType(
        {
            "engine": f"treemendous.applications.catalogs.{module}:{factory}",
            "example": f"examples/applications/catalogs/{artifact_name}.py",
            "oracle": f"tests/oracles/applications/catalogs/{artifact_name}.py",
            "benchmark": f"tests/performance/applications/catalogs/{artifact_name}.py",
            "docs": f"docs/scenarios/{scenario_id}.md",
        }
    )


EVIDENCE: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "genomic-annotation-overlap": _evidence(
            "genomic_annotation_overlap", "create_catalog", "genomic-annotation-overlap"
        ),
        "source-diagnostic-ranges": _evidence(
            "source_diagnostic_ranges", "create_catalog", "source-diagnostic-ranges"
        ),
        "filesystem-byte-locks": _evidence(
            "filesystem_byte_locks", "create_lock_table", "filesystem-byte-locks"
        ),
        "database-key-range-locks": _evidence(
            "database_key_range_locks",
            "create_lock_table",
            "database-key-range-locks",
        ),
        "packet-sequence-reassembly": _evidence(
            "packet_sequence_reassembly", "create_catalog", "packet-sequence-reassembly"
        ),
        "subtitle-cue-ranges": _evidence(
            "subtitle_cue_ranges", "create_catalog", "subtitle-cue-ranges"
        ),
        "video-edit-regions": _evidence(
            "video_edit_regions", "create_catalog", "video-edit-regions"
        ),
        "timeseries-alert-windows": _evidence(
            "timeseries_alert_windows", "create_catalog", "timeseries-alert-windows"
        ),
        "distributed-trace-spans": _evidence(
            "distributed_trace_spans", "create_catalog", "distributed-trace-spans"
        ),
        "morton-geospatial-ranges": _evidence(
            "morton_geospatial_ranges", "create_catalog", "morton-geospatial-ranges"
        ),
    }
)
