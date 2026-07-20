"""Attested benchmark for partitioned document search."""

from __future__ import annotations

from tests.oracles.applications.partitioning.document_search import (
    expected_hits,
    expected_index,
    tokenize,
)
from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from tests.performance.applications.partitioning._common import validate_case
from treemendous.applications.partitioning.document_search import (
    DocumentSearchEngine,
    SearchHit,
)

_DEFAULT_OPERATIONS = 240
_MAX_OPERATIONS = 1_000
_DEFAULT_SEED = 23
_SHARD_SIZE = 31


def run_benchmark(
    operations: int = _DEFAULT_OPERATIONS, seed: int = _DEFAULT_SEED
) -> ApplicationSample:
    """Search a deterministic bounded corpus and attest its complete index state."""
    validate_case(operations, seed, maximum=_MAX_OPERATIONS)
    selected_group = seed % 11
    documents = {
        index: (
            f"term group-{index % 11} bucket-{(index * 7 + seed) % 19} document-{index}"
        )
        for index in range(operations)
    }
    query = f"term group-{selected_group}"
    engine = DocumentSearchEngine(documents, query)

    def execute() -> tuple[SearchHit, ...]:
        return engine.run(shard_size=_SHARD_SIZE)

    def observe(raw: tuple[SearchHit, ...]) -> ApplicationOutcome:
        snapshot = engine.snapshot()
        hits = tuple((hit.document_id, hit.tokens) for hit in raw)
        state_hits = tuple((hit.document_id, hit.tokens) for hit in snapshot.hits)
        index = tuple(sorted(engine.index.items()))
        return ApplicationOutcome(
            results=hits,
            final_state={
                "query": snapshot.query,
                "hits": state_hits,
                "claimed_documents": snapshot.claimed_documents,
                "index": index,
            },
            counters={
                "documents_searched": snapshot.claimed_documents,
                "hits": len(raw),
                "search_bands": (operations + _SHARD_SIZE - 1) // _SHARD_SIZE,
                "run_calls": 1,
            },
        )

    def oracle() -> ApplicationOutcome:
        hit_ids = expected_hits(documents, query)
        hits = tuple(
            (document_id, tokenize(documents[document_id])) for document_id in hit_ids
        )
        return ApplicationOutcome(
            results=hits,
            final_state={
                "query": tuple(dict.fromkeys(tokenize(query))),
                "hits": hits,
                "claimed_documents": operations,
                "index": expected_index(documents),
            },
            counters={
                "documents_searched": operations,
                "hits": len(hits),
                "search_bands": (operations + _SHARD_SIZE - 1) // _SHARD_SIZE,
                "run_calls": 1,
            },
        )

    return run_application_case(
        scenario_id="partitioning.document_search",
        operations=operations,
        execute=execute,
        observe=observe,
        oracle=oracle,
    )


def run_smoke() -> int:
    """Run the default attested workload and return its operation count."""
    return run_benchmark().operations
