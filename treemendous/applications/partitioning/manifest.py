"""Data-only implementation evidence for the partitioning scenario family."""

from __future__ import annotations

from types import MappingProxyType

_PREFIX = "treemendous.applications.partitioning"
_EXAMPLE = "examples/one_dimensional/applications/partitioning"
_ORACLE = "tests/oracles/applications/partitioning"
_BENCHMARK = "tests/performance/applications/partitioning"
_DOCS = "docs/scenarios/partitioning"

_ROWS = (
    ("distributed-document-search", "document_search", "create_document_search"),
    ("distributed-regex-scan", "regex_scan", "create_regex_scan"),
    ("distributed-genetic-search", "genetic_search", "create_genetic_search"),
    ("distributed-graph-search", "graph_search", "create_graph_search"),
    ("distributed-sat-search", "sat_search", "create_sat_search"),
    ("distributed-fuzzing", "fuzzing", "create_fuzzing"),
    (
        "distributed-hyperparameter-search",
        "hyperparameter_search",
        "create_hyperparameter_search",
    ),
    ("distributed-log-replay", "log_replay", "create_log_replay"),
    ("distributed-build-sharding", "build_sharding", "create_build_sharding"),
    ("map-reduce-input-splits", "map_reduce", "create_map_reduce"),
    ("distributed-web-crawl", "web_crawl", "create_web_crawl"),
    ("distributed-index-merge", "index_merge", "create_index_merge"),
)

EVIDENCE = MappingProxyType(
    {
        scenario_id: MappingProxyType(
            {
                "engine": f"{_PREFIX}.{module}:{factory}",
                "example": f"{_EXAMPLE}/{module}.py",
                "oracle": f"{_ORACLE}/{module}.py",
                "benchmark": f"{_BENCHMARK}/{module}.py",
                "docs": f"{_DOCS}/{module}.md",
            }
        )
        for scenario_id, module, factory in _ROWS
    }
)

if set(EVIDENCE) != {row[0] for row in _ROWS} or len(EVIDENCE) != 12:
    raise RuntimeError("partitioning evidence must cover exactly twelve scenario IDs")
