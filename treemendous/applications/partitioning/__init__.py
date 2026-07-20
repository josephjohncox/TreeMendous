"""Reusable in-memory engines for the twelve partitioning/search scenarios.

Each engine models its named application semantics and uses private local
claim/event kernels. They do not claim to provide durable distributed
coordination; see each engine and scenario document for that boundary.
"""

from treemendous.applications.partitioning.build_sharding import (
    BuildShard,
    BuildShardingEngine,
    BuildTask,
    create_build_sharding,
)
from treemendous.applications.partitioning.document_search import (
    DocumentSearchEngine,
    SearchHit,
    create_document_search,
)
from treemendous.applications.partitioning.fuzzing import (
    Crash,
    FuzzingEngine,
    create_fuzzing,
)
from treemendous.applications.partitioning.genetic_search import (
    GeneticSearchEngine,
    create_genetic_search,
)
from treemendous.applications.partitioning.graph_search import (
    GraphSearchEngine,
    create_graph_search,
)
from treemendous.applications.partitioning.hyperparameter_search import (
    HyperparameterSearchEngine,
    TrialResult,
    create_hyperparameter_search,
)
from treemendous.applications.partitioning.index_merge import (
    IndexMergeEngine,
    TermPostings,
    create_index_merge,
)
from treemendous.applications.partitioning.log_replay import (
    LogReplayEngine,
    ReplayEvent,
    create_log_replay,
)
from treemendous.applications.partitioning.map_reduce import (
    InputSplit,
    MapReduceEngine,
    create_map_reduce,
)
from treemendous.applications.partitioning.regex_scan import (
    RegexMatch,
    RegexScanEngine,
    create_regex_scan,
)
from treemendous.applications.partitioning.sat_search import (
    SatisfyingAssignment,
    SatSearchEngine,
    create_sat_search,
)
from treemendous.applications.partitioning.web_crawl import (
    CrawlPage,
    WebCrawlEngine,
    create_web_crawl,
    normalize_url,
)

__all__ = [
    "BuildShard",
    "BuildShardingEngine",
    "BuildTask",
    "Crash",
    "CrawlPage",
    "DocumentSearchEngine",
    "FuzzingEngine",
    "GeneticSearchEngine",
    "GraphSearchEngine",
    "HyperparameterSearchEngine",
    "IndexMergeEngine",
    "InputSplit",
    "LogReplayEngine",
    "MapReduceEngine",
    "RegexMatch",
    "RegexScanEngine",
    "ReplayEvent",
    "SatSearchEngine",
    "SearchHit",
    "SatisfyingAssignment",
    "TermPostings",
    "TrialResult",
    "WebCrawlEngine",
    "create_build_sharding",
    "create_document_search",
    "create_fuzzing",
    "create_genetic_search",
    "create_graph_search",
    "create_hyperparameter_search",
    "create_index_merge",
    "create_log_replay",
    "create_map_reduce",
    "create_regex_scan",
    "create_sat_search",
    "create_web_crawl",
    "normalize_url",
]
