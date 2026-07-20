"""Correctness-checked smoke workload for document search."""

from tests.oracles.applications.partitioning.document_search import expected_hits
from treemendous.applications.partitioning.document_search import DocumentSearchEngine


def run_smoke() -> int:
    documents = {index: f"term group {index % 7}" for index in range(500)}
    engine = DocumentSearchEngine(documents, "term group")
    observed = tuple(hit.document_id for hit in engine.run(shard_size=31))
    expected = expected_hits(documents, "term group")
    if observed != expected:
        raise AssertionError("document-search smoke differs from oracle")
    return len(observed)
