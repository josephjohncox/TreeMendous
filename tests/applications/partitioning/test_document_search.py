"""Document-search engine contracts."""

import pytest

from tests.oracles.applications.partitioning.document_search import expected_hits
from treemendous.applications.partitioning.document_search import DocumentSearchEngine


def test_indexed_search_matches_independent_oracle_and_is_deterministic() -> None:
    documents = {9: "Blue range tree", 2: "tree only", 5: "RANGE blue blue"}
    engine = DocumentSearchEngine(documents, "blue range")
    observed = tuple(hit.document_id for hit in engine.run(shard_size=1))
    expected = (5, 9)
    assert observed == expected_hits(documents, "blue range") == expected
    assert engine.snapshot().claimed_documents == 3
    assert engine.run() == engine.snapshot().hits


def test_document_search_validates_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        DocumentSearchEngine({}, "term")
    with pytest.raises(ValueError, match="query"):
        DocumentSearchEngine({1: "text"}, "!!!")
