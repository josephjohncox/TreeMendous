"""Public adversarial coverage for partitioning application boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest

from treemendous.applications._shared.claiming import (
    ClaimInvariantError,
    ClaimState,
    ClaimUnavailableError,
    TerminalClaimError,
)
from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications.partitioning._runtime import (
    PartitionRuntime,
    RuntimeCheckpoint,
)
from treemendous.applications.partitioning.genetic_search import (
    GeneticGeneration,
    GeneticSearchEngine,
    create_genetic_search,
)
from treemendous.applications.partitioning.graph_search import (
    GraphSearchEngine,
    create_graph_search,
)
from treemendous.applications.partitioning.hyperparameter_search import (
    HyperparameterSearchEngine,
    create_hyperparameter_search,
)
from treemendous.applications.partitioning.index_merge import (
    IndexMergeEngine,
    create_index_merge,
)
from treemendous.applications.partitioning.log_replay import (
    LogReplayEngine,
    ReplayEvent,
    create_log_replay,
)
from treemendous.applications.partitioning.map_reduce import (
    MapReduceEngine,
    create_map_reduce,
)
from treemendous.applications.partitioning.web_crawl import (
    CrawlPage,
    WebCrawlEngine,
    create_web_crawl,
    normalize_url,
)


def _score_ones(value: str) -> float:
    return value.count("1") * 1.0


def _constant_objective(_: Mapping[str, Any]) -> float:
    return 1.0


def _pair_mapper(_: bytes) -> tuple[tuple[str, int], ...]:
    return (("key", 1),)


def _sum(left: int, right: int) -> int:
    return left + right


def test_runtime_transitions_and_callback_failures_are_atomic() -> None:
    runtime = PartitionRuntime(4)
    complete_claim = runtime.claim("complete", 1)
    assert runtime.validate_active(complete_claim) == complete_claim
    completed = runtime.complete(complete_claim, "done", {"count": 1})
    assert completed.state is ClaimState.COMPLETED
    with pytest.raises(TerminalClaimError):
        runtime.validate_active(complete_claim)

    committed: list[str] = []
    abandoned = runtime.abandon_claim(
        runtime.claim("abandon", 1), commit=lambda: committed.append("swapped")
    )
    assert abandoned.state is ClaimState.ABANDONED
    assert committed == ["swapped"]
    assert runtime.abandon(runtime.claim("plain-abandon", 1)).state is ClaimState.ABANDONED

    active = runtime.claim("retry", 1)
    with pytest.raises(RuntimeError, match="prepare failed"):
        runtime.execute_claim(
            active,
            kind="unused",
            prepare=lambda: (_ for _ in ()).throw(RuntimeError("prepare failed")),
            commit=lambda _: None,
            result=lambda _: None,
            abandon_on_error=False,
        )
    assert runtime.validate_active(active) == active


def test_runtime_commit_and_audit_callback_failures_preserve_runtime_state() -> None:
    runtime = PartitionRuntime(1)
    claim = runtime.claim("worker", 1)

    def failed_commit(_: str) -> None:
        raise RuntimeError("commit failed")

    with pytest.raises(RuntimeError, match="commit failed"):
        runtime.execute_claim(
            claim,
            kind="done",
            prepare=lambda: "prepared",
            commit=failed_commit,
            result=lambda _: {"ok": True},
        )
    checkpoint = runtime.checkpoint()
    assert checkpoint.claims.claims[0].state is ClaimState.ACTIVE
    no_events: tuple[object, ...] = ()
    assert checkpoint.events.events == no_events

    def failed_snapshot() -> None:
        raise RuntimeError("snapshot failed")

    with pytest.raises(RuntimeError, match="snapshot failed"):
        runtime.audit_snapshot(failed_snapshot)


def test_runtime_restore_rejects_cross_kernel_corruption() -> None:
    runtime = PartitionRuntime(1)
    claim = runtime.claim("worker", 1)
    runtime.complete(claim, "done")
    checkpoint = runtime.checkpoint()
    event = checkpoint.events.events[0]

    corruptions = (
        replace(checkpoint, events=replace(checkpoint.events, events=())),
        replace(
            checkpoint,
            events=replace(
                checkpoint.events,
                events=(replace(event, stream="invalid"),),
            ),
        ),
        replace(
            checkpoint,
            events=replace(
                checkpoint.events,
                events=(replace(event, stream="work:not-an-integer"),),
            ),
        ),
        replace(
            checkpoint,
            events=replace(
                checkpoint.events,
                events=(replace(event, stream="work:99"),),
            ),
        ),
        replace(
            checkpoint,
            events=replace(
                checkpoint.events,
                events=(replace(event, payload=(("start", 99), ("end", 100))),),
            ),
        ),
        replace(
            checkpoint,
            events=replace(
                checkpoint.events,
                events=(
                    replace(
                        event,
                        payload=(
                            ("start", claim.span.start),
                            ("end", claim.span.end),
                            ("fencing_token", claim.fencing_token + 1),
                        ),
                    ),
                ),
            ),
        ),
    )
    for corrupted in corruptions:
        with pytest.raises(ClaimInvariantError):
            PartitionRuntime.from_checkpoint(corrupted, clock=LogicalClock())

    with pytest.raises(TypeError, match="RuntimeCheckpoint"):
        PartitionRuntime.from_checkpoint(object(), clock=LogicalClock())  # type: ignore[arg-type]
    restored = PartitionRuntime.from_checkpoint(checkpoint, clock=LogicalClock())
    assert restored.checkpoint() == checkpoint

    abandoned_runtime = PartitionRuntime(1)
    abandoned_runtime.abandon(abandoned_runtime.claim("worker", 1))
    abandoned_checkpoint = abandoned_runtime.checkpoint()
    abandoned_event = abandoned_checkpoint.events.events[0]
    wrong_abandonment = replace(
        abandoned_checkpoint,
        events=replace(
            abandoned_checkpoint.events,
            events=(replace(abandoned_event, kind="done"),),
        ),
    )
    with pytest.raises(ClaimInvariantError, match="abandonment"):
        PartitionRuntime.from_checkpoint(wrong_abandonment, clock=LogicalClock())


def test_url_normalization_and_crawler_validate_adversarial_inputs() -> None:
    invalid_urls = (
        "",
        "ftp://example.test/path",
        "http://example.test:invalid/",
        "http://user:password@example.test/",
    )
    for url in invalid_urls:
        with pytest.raises(ValueError):
            normalize_url(url)

    assert normalize_url("https://EXAMPLE.test:8443/a/../b/?z=&a=2") == (
        "https://example.test:8443/b/?a=2&z="
    )
    assert normalize_url("child", base="https://example.test/root/") == (
        "https://example.test/root/child"
    )

    def fetcher(_: str) -> CrawlPage:
        return CrawlPage(b"ok", ())

    invalid_constructors = (
        lambda: WebCrawlEngine("https://example.test", fetcher, max_pages=1),
        lambda: WebCrawlEngine((), fetcher, max_pages=1),
        lambda: WebCrawlEngine(
            ("https://example.test",), None, max_pages=1  # type: ignore[arg-type]
        ),
        lambda: WebCrawlEngine(("https://example.test",), fetcher, max_pages=0),
    )
    for construct in invalid_constructors:
        with pytest.raises((TypeError, ValueError)):
            construct()


def test_crawler_rejects_invalid_fetch_results_and_stops_at_budget() -> None:
    def return_object(_: str) -> Any:
        return object()

    def return_invalid_body(_: str) -> CrawlPage:
        return CrawlPage("text", ())  # type: ignore[arg-type]

    invalid_fetchers = (return_object, return_invalid_body)
    for invalid_fetcher in invalid_fetchers:
        engine = WebCrawlEngine(
            ("https://example.test",), invalid_fetcher, max_pages=1
        )
        with pytest.raises(RuntimeError, match="fetch failed"):
            engine.crawl_next()
        no_visits: tuple[str, ...] = ()
        assert engine.snapshot().visited == no_visits

    engine = WebCrawlEngine(
        ("https://example.test",),
        lambda _: CrawlPage(b"root", ("/next",)),
        max_pages=1,
    )
    assert engine.crawl_next() == "https://example.test/"
    assert engine.crawl_next() is None
    snapshot, runtime = engine.audit_snapshot()
    expected_visit = ("https://example.test/",)
    assert snapshot.visited == expected_visit
    assert isinstance(runtime, RuntimeCheckpoint)
    assert runtime.claims.claims[0].state is ClaimState.COMPLETED
    expected_fixture_visits = (
        "https://example.invalid/",
        "https://example.invalid/about",
    )
    assert create_web_crawl(max_pages=2).run().visited == expected_fixture_visits


def test_hyperparameter_validators_and_minimization_paths() -> None:
    invalid_constructors = (
        lambda: HyperparameterSearchEngine({}, _constant_objective),
        lambda: HyperparameterSearchEngine(
            {"x": (1,)}, None  # type: ignore[arg-type]
        ),
        lambda: HyperparameterSearchEngine(
            {"x": (1,)}, _constant_objective, maximize=1  # type: ignore[arg-type]
        ),
        lambda: HyperparameterSearchEngine(
            {1: (1,)}, _constant_objective  # type: ignore[dict-item]
        ),
        lambda: HyperparameterSearchEngine({"": (1,)}, _constant_objective),
        lambda: HyperparameterSearchEngine({"x": "one"}, _constant_objective),
    )
    for construct in invalid_constructors:
        with pytest.raises((TypeError, ValueError)):
            construct()

    def numeric_objective(parameters: Mapping[str, Any]) -> float:
        value = parameters["x"]
        if type(value) is not int:
            raise TypeError("x must be an integer")
        return value * 1.0

    engine = HyperparameterSearchEngine(
        {"x": (2, 1)}, numeric_objective, maximize=False
    )
    for trial_id in (-1, True, 2):
        with pytest.raises(ValueError, match="outside"):
            engine.parameters_for(trial_id)
    expected_minimized_scores = (1.0, 2.0)
    assert tuple(result.score for result in engine.run()) == expected_minimized_scores
    snapshot, runtime = engine.audit_snapshot()
    assert snapshot.ranking == engine.ranking()
    assert isinstance(runtime, RuntimeCheckpoint)
    assert len(runtime.events.events) == 1

    defaults = create_hyperparameter_search(
        {"value": (True, "ignored", 2, 0.5)}, maximize=True
    ).run()
    expected_default_scores = (2.0, 0.5, 0.0, 0.0)
    assert tuple(result.score for result in defaults) == expected_default_scores
    assert len(create_hyperparameter_search().run()) == 4


def test_hyperparameter_objective_conversion_failure_is_atomic() -> None:
    def failed(_: Mapping[str, Any]) -> object:
        raise OSError("objective unavailable")

    engine = HyperparameterSearchEngine({"x": (1,)}, failed)  # type: ignore[arg-type]
    claim = engine.claim("worker", 1)
    with pytest.raises(ValueError, match="finite"):
        engine.evaluate_claim(claim)
    no_results: tuple[object, ...] = ()
    assert engine.ranking() == no_results


def test_genetic_validators_callback_failures_and_terminal_step() -> None:
    invalid_constructors = (
        lambda: GeneticSearchEngine("01", _score_ones, generations=1),
        lambda: GeneticSearchEngine(("", ""), _score_ones, generations=1),
        lambda: GeneticSearchEngine(("0", "01"), _score_ones, generations=1),
        lambda: GeneticSearchEngine(("0", "x"), _score_ones, generations=1),
        lambda: GeneticSearchEngine(
            ("0", "1"), None, generations=1  # type: ignore[arg-type]
        ),
        lambda: GeneticSearchEngine(("0", "1"), _score_ones, generations=0),
        lambda: GeneticSearchEngine(("0", "1"), _score_ones, generations=1, seed=True),
        lambda: GeneticSearchEngine(
            ("0", "1"), _score_ones, generations=1, mutation_rate=True
        ),
    )
    for construct in invalid_constructors:
        with pytest.raises((TypeError, ValueError)):
            construct()

    def failed_fitness(_: str) -> float:
        raise OSError("fitness unavailable")

    failed = GeneticSearchEngine(("0", "1"), failed_fitness, generations=1)
    with pytest.raises(ValueError, match="fitness evaluation"):
        failed.step()
    assert failed.checkpoint().generation == 0

    engine = GeneticSearchEngine(
        ("0", "1"), _score_ones, generations=1, mutation_rate=1.0
    )
    expected_best = (1.0, "1")
    assert engine.best() == expected_best
    engine.step()
    with pytest.raises(ClaimUnavailableError, match="complete"):
        engine.step()
    assert create_genetic_search(generations=1).run()[0].number == 0
    assert create_genetic_search(fitness=_score_ones, generations=1).run()[0].number == 0


def test_genetic_restore_rejects_application_checkpoint_corruption() -> None:
    clock = LogicalClock()
    engine = GeneticSearchEngine(("00", "11"), _score_ones, generations=2, clock=clock)
    engine.step()
    checkpoint = engine.checkpoint()
    record = checkpoint.history[0]

    corruptions: tuple[tuple[object, type[Exception], str], ...] = (
        (object(), TypeError, "GeneticCheckpoint"),
        (replace(checkpoint, generations=0), ValueError, "greater than zero"),
        (replace(checkpoint, generation=True), ValueError, "generation"),
        (
            replace(checkpoint, history=[]),  # type: ignore[arg-type]
            ValueError,
            "history",
        ),
        (
            replace(checkpoint, history=(object(),)),  # type: ignore[arg-type]
            ValueError,
            "numbering",
        ),
        (
            replace(checkpoint, history=(replace(record, number=1),)),
            ValueError,
            "numbering",
        ),
        (
            replace(checkpoint, history=(replace(record, population=("0", "1")),)),
            ValueError,
            "population",
        ),
        (
            replace(checkpoint, history=(replace(record, ranking=()),)),
            ValueError,
            "population",
        ),
        (replace(checkpoint, random_state=object()), ValueError, "random state"),
        (replace(checkpoint, generations=3), ClaimInvariantError, "domain"),
    )
    for corrupted, error, message in corruptions:
        with pytest.raises(error, match=message):
            GeneticSearchEngine.from_checkpoint(
                corrupted,  # type: ignore[arg-type]
                fitness=_score_ones,
                clock=clock,
            )

    with pytest.raises(TypeError, match="callable"):
        GeneticSearchEngine.from_checkpoint(checkpoint, fitness=None, clock=clock)  # type: ignore[arg-type]


def test_genetic_restore_rejects_unfinished_and_malformed_runtime_claims() -> None:
    clock = LogicalClock()
    base_engine = GeneticSearchEngine(("00", "11"), _score_ones, generations=2)
    base_checkpoint = base_engine.checkpoint()

    active_runtime = PartitionRuntime(2, clock=clock)
    active_runtime.claim("worker", 1)
    unfinished = replace(base_checkpoint, runtime=active_runtime.checkpoint())
    with pytest.raises(ClaimInvariantError, match="unfinished"):
        GeneticSearchEngine.from_checkpoint(
            unfinished, fitness=_score_ones, clock=clock
        )

    history = (
        GeneticGeneration(0, ("00", "11"), ((2.0, "11"), (0.0, "00"))),
    )
    metadata_runtime = PartitionRuntime(2, clock=clock)
    claim = metadata_runtime.claim("worker", 1)
    metadata_runtime.complete(claim, "generation", {"other": 0})
    malformed_metadata = replace(
        base_checkpoint,
        generation=1,
        history=history,
        runtime=metadata_runtime.checkpoint(),
    )
    with pytest.raises(ClaimInvariantError, match="metadata"):
        GeneticSearchEngine.from_checkpoint(
            malformed_metadata, fitness=_score_ones, clock=clock
        )

    wide_runtime = PartitionRuntime(2, clock=clock)
    wide_claim = wide_runtime.claim("worker", 2)
    wide_runtime.complete(wide_claim, "generation", {"generation": 0})
    malformed_span = replace(
        base_checkpoint,
        generation=1,
        history=history,
        runtime=wide_runtime.checkpoint(),
    )
    with pytest.raises(ClaimInvariantError, match="claims"):
        GeneticSearchEngine.from_checkpoint(
            malformed_span, fitness=_score_ones, clock=clock
        )


def test_log_replay_rejects_every_invalid_event_shape() -> None:
    invalid_events: tuple[object, ...] = (
        "events",
        (),
        (object(),),
        (ReplayEvent(-1, "x", "set", 1),),
        (ReplayEvent(0, "x", "set", 1), ReplayEvent(0, "y", "set", 2)),
        (ReplayEvent(0, "x", "unknown", 1),),
        (ReplayEvent(0, "x", "set", None),),
        (ReplayEvent(0, "x", "increment", "1"),),
        (ReplayEvent(0, "x", "delete", 1),),
    )
    for events in invalid_events:
        with pytest.raises((TypeError, ValueError)):
            LogReplayEngine(events)

    with pytest.raises((TypeError, ValueError), match="event key"):
        LogReplayEngine((ReplayEvent(0, "", "delete"),))
    expected_state = (("count", 3),)
    assert create_log_replay().run() == expected_state


def test_graph_search_validators_empty_frontier_and_factory_paths() -> None:
    class UnavailableGraphSearch(GraphSearchEngine):
        def expand(
            self, *, width: int = 1, owner: str = "local"
        ) -> tuple[str, ...]:
            del width, owner
            raise ClaimUnavailableError("frontier temporarily unavailable")

    invalid_graphs: tuple[tuple[object, object], ...] = (
        ({}, "a"),
        ({"a": "b", "b": ()}, "a"),
        ({"": ()}, ""),
        ({"a": (1,)}, "a"),
        ({"a": ()}, "missing"),
    )
    for graph, start in invalid_graphs:
        with pytest.raises((TypeError, ValueError)):
            GraphSearchEngine(graph, start)  # type: ignore[arg-type]

    engine = GraphSearchEngine({"a": ()}, "a")
    expected_single = ("a",)
    no_vertices: tuple[str, ...] = ()
    assert engine.expand() == expected_single
    assert engine.expand() == no_vertices
    snapshot, runtime = engine.audit_snapshot()
    assert snapshot.order == expected_single
    assert isinstance(runtime, RuntimeCheckpoint)
    assert len(runtime.events.events) == 1
    expected_default_order = ("a", "b", "c", "d")
    assert create_graph_search().run().order == expected_default_order
    expected_custom_order = ("z",)
    custom_graph: Mapping[str, tuple[str, ...]] = {"z": ()}
    assert create_graph_search(custom_graph, "z").run().order == expected_custom_order

    unavailable = UnavailableGraphSearch({"a": ()}, "a").run()
    assert unavailable.frontier == expected_single


def test_index_merge_validators_audit_and_factory_paths() -> None:
    invalid_segments: tuple[object, ...] = (
        "segment",
        (),
        (object(),),
        ({1: (1,)},),
        ({"": (1,)},),
        ({"term": "posting"},),
        ({},),
    )
    for segments in invalid_segments:
        with pytest.raises((TypeError, ValueError)):
            IndexMergeEngine(segments)

    engine = IndexMergeEngine(({"term": (1,)},))
    engine.run()
    snapshot, runtime = engine.audit_snapshot()
    expected_postings = (1,)
    assert snapshot.merged[0].postings == expected_postings
    assert isinstance(runtime, RuntimeCheckpoint)
    assert len(runtime.events.events) == 1
    assert len(create_index_merge().run()) == 3


def test_map_reduce_validators_emissions_and_default_factory() -> None:
    invalid_constructors = (
        lambda: MapReduceEngine(b"", _pair_mapper, _sum, split_size=1),
        lambda: MapReduceEngine(
            b"x", None, _sum, split_size=1  # type: ignore[arg-type]
        ),
        lambda: MapReduceEngine(
            b"x", _pair_mapper, None, split_size=1  # type: ignore[arg-type]
        ),
        lambda: MapReduceEngine(b"x", _pair_mapper, _sum, split_size=0),
        lambda: MapReduceEngine(b"x", _pair_mapper, _sum, split_size=1, mode="bad"),
    )
    for construct in invalid_constructors:
        with pytest.raises((TypeError, ValueError)):
            construct()

    def invalid_mapper(_: bytes) -> tuple[str, ...]:
        return ("not-a-pair",)

    engine = MapReduceEngine(b"x", invalid_mapper, _sum, split_size=1)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="mapper"):
        engine.run()
    no_results: tuple[object, ...] = ()
    assert engine.snapshot().results == no_results

    default = create_map_reduce()
    assert default.splits[0].start == 0
    expected_counts = (("range", 2), ("sets", 1), ("trees", 1))
    assert default.run() == expected_counts
    assert create_map_reduce(data=b" ", mode="bytes").run() == no_results
