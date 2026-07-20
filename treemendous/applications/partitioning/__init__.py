"""Reusable in-memory engines for the twelve partitioning/search scenarios.

Exports are resolved lazily so importing the data-only family manifest does not
also import every application engine.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from treemendous.applications.partitioning.build_sharding import (
        BuildShard as BuildShard,
    )
    from treemendous.applications.partitioning.build_sharding import (
        BuildShardingEngine as BuildShardingEngine,
    )
    from treemendous.applications.partitioning.build_sharding import (
        BuildTask as BuildTask,
    )
    from treemendous.applications.partitioning.build_sharding import (
        create_build_sharding as create_build_sharding,
    )
    from treemendous.applications.partitioning.document_search import (
        DocumentSearchEngine as DocumentSearchEngine,
    )
    from treemendous.applications.partitioning.document_search import (
        SearchHit as SearchHit,
    )
    from treemendous.applications.partitioning.document_search import (
        create_document_search as create_document_search,
    )
    from treemendous.applications.partitioning.fuzzing import Crash as Crash
    from treemendous.applications.partitioning.fuzzing import (
        FuzzingEngine as FuzzingEngine,
    )
    from treemendous.applications.partitioning.fuzzing import (
        create_fuzzing as create_fuzzing,
    )
    from treemendous.applications.partitioning.genetic_search import (
        GeneticSearchEngine as GeneticSearchEngine,
    )
    from treemendous.applications.partitioning.genetic_search import (
        create_genetic_search as create_genetic_search,
    )
    from treemendous.applications.partitioning.graph_search import (
        GraphSearchEngine as GraphSearchEngine,
    )
    from treemendous.applications.partitioning.graph_search import (
        create_graph_search as create_graph_search,
    )
    from treemendous.applications.partitioning.hyperparameter_search import (
        HyperparameterSearchEngine as HyperparameterSearchEngine,
    )
    from treemendous.applications.partitioning.hyperparameter_search import (
        TrialResult as TrialResult,
    )
    from treemendous.applications.partitioning.hyperparameter_search import (
        create_hyperparameter_search as create_hyperparameter_search,
    )
    from treemendous.applications.partitioning.index_merge import (
        IndexMergeEngine as IndexMergeEngine,
    )
    from treemendous.applications.partitioning.index_merge import (
        TermPostings as TermPostings,
    )
    from treemendous.applications.partitioning.index_merge import (
        create_index_merge as create_index_merge,
    )
    from treemendous.applications.partitioning.log_replay import (
        LogReplayEngine as LogReplayEngine,
    )
    from treemendous.applications.partitioning.log_replay import (
        ReplayEvent as ReplayEvent,
    )
    from treemendous.applications.partitioning.log_replay import (
        create_log_replay as create_log_replay,
    )
    from treemendous.applications.partitioning.map_reduce import (
        InputSplit as InputSplit,
    )
    from treemendous.applications.partitioning.map_reduce import (
        MapReduceEngine as MapReduceEngine,
    )
    from treemendous.applications.partitioning.map_reduce import (
        create_map_reduce as create_map_reduce,
    )
    from treemendous.applications.partitioning.regex_scan import (
        RegexMatch as RegexMatch,
    )
    from treemendous.applications.partitioning.regex_scan import (
        RegexScanEngine as RegexScanEngine,
    )
    from treemendous.applications.partitioning.regex_scan import (
        create_regex_scan as create_regex_scan,
    )
    from treemendous.applications.partitioning.sat_search import (
        SatisfyingAssignment as SatisfyingAssignment,
    )
    from treemendous.applications.partitioning.sat_search import (
        SatSearchEngine as SatSearchEngine,
    )
    from treemendous.applications.partitioning.sat_search import (
        create_sat_search as create_sat_search,
    )
    from treemendous.applications.partitioning.web_crawl import (
        CrawlPage as CrawlPage,
    )
    from treemendous.applications.partitioning.web_crawl import (
        WebCrawlEngine as WebCrawlEngine,
    )
    from treemendous.applications.partitioning.web_crawl import (
        create_web_crawl as create_web_crawl,
    )
    from treemendous.applications.partitioning.web_crawl import (
        normalize_url as normalize_url,
    )

_EXPORTS = {
    "BuildShard": ("build_sharding", "BuildShard"),
    "BuildShardingEngine": ("build_sharding", "BuildShardingEngine"),
    "BuildTask": ("build_sharding", "BuildTask"),
    "Crash": ("fuzzing", "Crash"),
    "CrawlPage": ("web_crawl", "CrawlPage"),
    "DocumentSearchEngine": ("document_search", "DocumentSearchEngine"),
    "FuzzingEngine": ("fuzzing", "FuzzingEngine"),
    "GeneticSearchEngine": ("genetic_search", "GeneticSearchEngine"),
    "GraphSearchEngine": ("graph_search", "GraphSearchEngine"),
    "HyperparameterSearchEngine": (
        "hyperparameter_search",
        "HyperparameterSearchEngine",
    ),
    "IndexMergeEngine": ("index_merge", "IndexMergeEngine"),
    "InputSplit": ("map_reduce", "InputSplit"),
    "LogReplayEngine": ("log_replay", "LogReplayEngine"),
    "MapReduceEngine": ("map_reduce", "MapReduceEngine"),
    "RegexMatch": ("regex_scan", "RegexMatch"),
    "RegexScanEngine": ("regex_scan", "RegexScanEngine"),
    "ReplayEvent": ("log_replay", "ReplayEvent"),
    "SatSearchEngine": ("sat_search", "SatSearchEngine"),
    "SearchHit": ("document_search", "SearchHit"),
    "SatisfyingAssignment": ("sat_search", "SatisfyingAssignment"),
    "TermPostings": ("index_merge", "TermPostings"),
    "TrialResult": ("hyperparameter_search", "TrialResult"),
    "WebCrawlEngine": ("web_crawl", "WebCrawlEngine"),
    "create_build_sharding": ("build_sharding", "create_build_sharding"),
    "create_document_search": ("document_search", "create_document_search"),
    "create_fuzzing": ("fuzzing", "create_fuzzing"),
    "create_genetic_search": ("genetic_search", "create_genetic_search"),
    "create_graph_search": ("graph_search", "create_graph_search"),
    "create_hyperparameter_search": (
        "hyperparameter_search",
        "create_hyperparameter_search",
    ),
    "create_index_merge": ("index_merge", "create_index_merge"),
    "create_log_replay": ("log_replay", "create_log_replay"),
    "create_map_reduce": ("map_reduce", "create_map_reduce"),
    "create_regex_scan": ("regex_scan", "create_regex_scan"),
    "create_sat_search": ("sat_search", "create_sat_search"),
    "create_web_crawl": ("web_crawl", "create_web_crawl"),
    "normalize_url": ("web_crawl", "normalize_url"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve one public engine export without eager family imports."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, attribute)
    globals()[name] = value
    return value
