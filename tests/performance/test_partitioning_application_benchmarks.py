"""Attestation coverage for every partitioning application benchmark."""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType

import pytest

from tests.performance.applications.harness import ApplicationSample

_SCENARIO_IDS = {
    "build_sharding": "distributed-build-sharding",
    "document_search": "distributed-document-search",
    "fuzzing": "distributed-fuzzing",
    "genetic_search": "distributed-genetic-search",
    "graph_search": "distributed-graph-search",
    "hyperparameter_search": "distributed-hyperparameter-search",
    "index_merge": "distributed-index-merge",
    "log_replay": "distributed-log-replay",
    "map_reduce": "map-reduce-input-splits",
    "regex_scan": "distributed-regex-scan",
    "sat_search": "distributed-sat-search",
    "web_crawl": "distributed-web-crawl",
}
_MODULE_NAMES = tuple(_SCENARIO_IDS)
_EXPECTED_PARAMETERS = ("operations", "seed")


def _module(name: str) -> ModuleType:
    return importlib.import_module(
        f"tests.performance.applications.partitioning.{name}"
    )


@pytest.mark.parametrize("name", _MODULE_NAMES)
def test_partitioning_run_benchmark_contract(name: str) -> None:
    """Every scenario accepts uniform keywords and returns validated evidence."""
    module = _module(name)
    signature = inspect.signature(module.run_benchmark)
    parameter_names = tuple(signature.parameters)
    assert parameter_names == _EXPECTED_PARAMETERS

    sample = module.run_benchmark(operations=8, seed=101)
    repeated = module.run_benchmark(operations=8, seed=101)

    assert isinstance(sample, ApplicationSample)
    assert sample.scenario_id == _SCENARIO_IDS[name]
    assert sample.operations == 8
    assert sample.execution_ns >= 0
    assert sample.validated
    assert sample.result_checksum == repeated.result_checksum
    assert sample.state_checksum == repeated.state_checksum
    assert sample.counters_checksum == repeated.counters_checksum
    assert sample.evidence_checksum == repeated.evidence_checksum


@pytest.mark.parametrize("name", _MODULE_NAMES)
def test_partitioning_run_benchmark_rejects_unbounded_work(name: str) -> None:
    """Every scenario rejects a workload beyond the family-wide safe ceiling."""
    module = _module(name)

    with pytest.raises(ValueError, match="must not exceed"):
        module.run_benchmark(operations=10_000, seed=101)


def test_document_search_detects_result_sequence_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupted independent result oracle cannot produce a sample."""
    module = _module("document_search")
    monkeypatch.setattr(module, "expected_hits", lambda _documents, _query: ())

    with pytest.raises(AssertionError, match="evidence differs"):
        module.run_benchmark(operations=8, seed=7)


def test_web_crawl_detects_final_state_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupted independent final-state oracle cannot produce a sample."""
    module = _module("web_crawl")
    monkeypatch.setattr(
        module,
        "expected_frontier",
        lambda _seed, _links, _limit: ("https://crawl.test/wrong",),
    )

    with pytest.raises(AssertionError, match="evidence differs"):
        module.run_benchmark(operations=8, seed=7)
