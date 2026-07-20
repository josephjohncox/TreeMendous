"""Adversarial identity, fencing, atomicity, and snapshot contracts."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import replace
from threading import Event as ThreadEvent
from threading import Thread
from typing import Any

import pytest

from treemendous.applications._shared.claiming import (
    ClaimInvariantError,
    ClaimState,
    ForeignClaimError,
    StaleClaimError,
    TerminalClaimError,
)
from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications._shared.events import EventLog
from treemendous.applications.partitioning._runtime import (
    PartitionRuntime,
    RuntimeCheckpoint,
)
from treemendous.applications.partitioning.build_sharding import (
    BuildShardingEngine,
    BuildTask,
)
from treemendous.applications.partitioning.document_search import DocumentSearchEngine
from treemendous.applications.partitioning.fuzzing import FuzzingEngine
from treemendous.applications.partitioning.hyperparameter_search import (
    HyperparameterSearchEngine,
)
from treemendous.applications.partitioning.index_merge import IndexMergeEngine
from treemendous.applications.partitioning.log_replay import (
    LogReplayEngine,
    ReplayEvent,
)
from treemendous.applications.partitioning.map_reduce import MapReduceEngine
from treemendous.applications.partitioning.regex_scan import RegexScanEngine
from treemendous.applications.partitioning.sat_search import SatSearchEngine

EngineFactory = Callable[[list[str]], Any]
Executor = Callable[[Any, Any], object]


def _build(_: list[str]) -> BuildShardingEngine:
    return BuildShardingEngine((BuildTask("compile"),), shard_count=1)


def _document(_: list[str]) -> DocumentSearchEngine:
    return DocumentSearchEngine({0: "range"}, "range")


def _fuzz(effects: list[str]) -> FuzzingEngine:
    def target(_: bytes) -> None:
        effects.append("target")

    return FuzzingEngine(target, cases=1)


def _hyperparameter(effects: list[str]) -> HyperparameterSearchEngine:
    def objective(_: Mapping[str, Any]) -> float:
        effects.append("objective")
        return 1.0

    return HyperparameterSearchEngine({"x": (1,)}, objective)


def _index(_: list[str]) -> IndexMergeEngine:
    return IndexMergeEngine(({"term": (1,)},))


def _log(_: list[str]) -> LogReplayEngine:
    return LogReplayEngine((ReplayEvent(0, "key", "set", 1),))


def _map_reduce(effects: list[str]) -> MapReduceEngine:
    def mapper(_: bytes) -> tuple[tuple[str, int], ...]:
        effects.append("mapper")
        return (("key", 1),)

    return MapReduceEngine(
        b"unit", mapper, lambda left, right: left + right, split_size=1
    )


def _regex(_: list[str]) -> RegexScanEngine:
    return RegexScanEngine(b"a", b"a", halo=0)


def _sat(_: list[str]) -> SatSearchEngine:
    return SatSearchEngine(1, ((1,),), prefix_bits=0)


_EXECUTORS: tuple[tuple[str, EngineFactory, Executor], ...] = (
    ("build", _build, lambda engine, claim: engine.execute_claim(claim)),
    ("document", _document, lambda engine, claim: engine.search_claim(claim)),
    ("fuzz", _fuzz, lambda engine, claim: engine.execute_claim(claim)),
    (
        "hyperparameter",
        _hyperparameter,
        lambda engine, claim: engine.evaluate_claim(claim),
    ),
    ("index", _index, lambda engine, claim: engine.merge_claim(claim)),
    ("log", _log, lambda engine, claim: engine.apply_claim(claim)),
    ("map-reduce", _map_reduce, lambda engine, claim: engine.execute_claim(claim)),
    ("regex", _regex, lambda engine, claim: engine.scan_claim(claim)),
    ("sat", _sat, lambda engine, claim: engine.evaluate_claim(claim)),
)


@pytest.mark.parametrize(("name", "factory", "execute"), _EXECUTORS)
def test_explicit_executors_reject_invalid_claims_before_effects(
    name: str, factory: EngineFactory, execute: Executor
) -> None:
    del name
    effects: list[str] = []
    engine = factory(effects)
    foreign = factory([])
    claim = engine.claim("worker", 1)
    foreign_claim = foreign.claim("worker", 1)
    initial = engine.snapshot()

    with pytest.raises(ForeignClaimError):
        execute(engine, foreign_claim)
    assert engine.snapshot() == initial
    assert effects == []

    stale = replace(claim, fencing_token=claim.fencing_token + 10)
    with pytest.raises(StaleClaimError):
        execute(engine, stale)
    assert engine.snapshot() == initial
    assert effects == []

    execute(engine, claim)
    terminal = engine.snapshot()
    effects_after_completion = tuple(effects)
    with pytest.raises(TerminalClaimError):
        execute(engine, claim)
    assert engine.snapshot() == terminal
    assert tuple(effects) == effects_after_completion


def _completed_runtime_checkpoint() -> RuntimeCheckpoint:
    runtime = PartitionRuntime(2)
    first = runtime.claim("worker", 1)
    second = runtime.claim("worker", 1)
    runtime.complete(first, "first")
    runtime.complete(second, "second")
    return runtime.checkpoint()


def test_runtime_restore_rejects_duplicate_and_missing_claim_lineage() -> None:
    checkpoint = _completed_runtime_checkpoint()
    first_event, second_event = checkpoint.events.events
    forged_event = replace(
        second_event,
        stream=first_event.stream,
        version=2,
    )
    forged_requests = tuple(
        replace(request, stream=first_event.stream, expected_version=1)
        if request.event_sequence == second_event.sequence
        else request
        for request in checkpoint.events.requests
    )
    forged = replace(
        checkpoint,
        events=replace(
            checkpoint.events,
            events=(first_event, forged_event),
            requests=forged_requests,
        ),
    )

    with pytest.raises(ClaimInvariantError, match="exactly once"):
        PartitionRuntime.from_checkpoint(forged, clock=LogicalClock())


def test_runtime_restore_rejects_reordered_terminal_transition_ids() -> None:
    checkpoint = _completed_runtime_checkpoint()
    first_event, second_event = checkpoint.events.events
    forged_events = (
        replace(second_event, sequence=1),
        replace(first_event, sequence=2),
    )
    forged_requests = tuple(
        replace(request, event_sequence=3 - request.event_sequence)
        for request in checkpoint.events.requests
    )
    forged = replace(
        checkpoint,
        events=replace(
            checkpoint.events,
            events=forged_events,
            requests=forged_requests,
        ),
    )

    with pytest.raises(ClaimInvariantError, match="terminal claim order"):
        PartitionRuntime.from_checkpoint(forged, clock=LogicalClock())


def test_failed_runtime_event_append_cannot_commit_application_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = DocumentSearchEngine({0: "range"}, "range")
    claim = engine.claim("worker", 1)
    original_append = EventLog.append

    def fail_work_event(self: EventLog, stream: str, *args: Any, **kwargs: Any) -> Any:
        if stream.startswith("work:"):
            raise RuntimeError("injected event failure")
        return original_append(self, stream, *args, **kwargs)

    monkeypatch.setattr(EventLog, "append", fail_work_event)
    with pytest.raises(RuntimeError, match="injected event failure"):
        engine.search_claim(claim)

    application, runtime = engine.audit_snapshot()
    assert isinstance(runtime, RuntimeCheckpoint)
    empty: tuple[object, ...] = ()
    assert application.hits == empty
    assert runtime.events.events == empty
    assert runtime.claims.claims[0].state is ClaimState.ACTIVE


def test_concurrent_duplicate_execution_runs_callback_once_and_snapshot_is_consistent() -> (
    None
):
    entered = ThreadEvent()
    release = ThreadEvent()
    callback_calls: list[str] = []

    def mapper(_: bytes) -> tuple[tuple[str, int], ...]:
        callback_calls.append("mapper")
        entered.set()
        assert release.wait(timeout=5)
        return (("key", 1),)

    engine = MapReduceEngine(
        b"unit", mapper, lambda left, right: left + right, split_size=1
    )
    claim = engine.claim("worker", 1)
    outcomes: list[object] = []

    def execute() -> None:
        try:
            outcomes.append(engine.execute_claim(claim))
        except Exception as exc:  # test captures the losing fenced execution
            outcomes.append(exc)

    audit_finished = ThreadEvent()
    audits: list[tuple[Any, Any]] = []

    def audit() -> None:
        audits.append(engine.audit_snapshot())
        audit_finished.set()

    first = Thread(target=execute)
    second = Thread(target=execute)
    observer = Thread(target=audit)
    first.start()
    assert entered.wait(timeout=5)
    second.start()
    observer.start()
    assert not audit_finished.wait(timeout=0.05)
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)
    observer.join(timeout=5)

    assert audit_finished.is_set()
    application, runtime = audits[0]
    assert isinstance(runtime, RuntimeCheckpoint)
    assert callback_calls == ["mapper"]
    expected_results = (("key", 1),)
    assert application.results == expected_results
    assert len(runtime.events.events) == 1
    assert runtime.claims.claims[0].state is ClaimState.COMPLETED
    assert sum(isinstance(item, TerminalClaimError) for item in outcomes) == 1


def test_reentrant_execution_is_fenced_before_a_second_callback() -> None:
    callback_calls = 0
    rejected: list[Exception] = []
    engine: MapReduceEngine
    claim: Any

    def mapper(_: bytes) -> tuple[tuple[str, int], ...]:
        nonlocal callback_calls
        callback_calls += 1
        try:
            engine.execute_claim(claim)
        except Exception as exc:
            rejected.append(exc)
        return (("key", 1),)

    engine = MapReduceEngine(
        b"unit", mapper, lambda left, right: left + right, split_size=1
    )
    claim = engine.claim("worker", 1)
    result = engine.execute_claim(claim)
    expected_result = (("key", 1),)
    assert result == expected_result
    assert callback_calls == 1
    assert len(rejected) == 1
    assert isinstance(rejected[0], StaleClaimError)


@pytest.mark.parametrize(
    ("data", "pattern", "flags"),
    (
        (b"xxxabc", b"^abc", 0),
        (b"abcxxx", b"abc$", 0),
        (b"xxx\nabc", b"^abc", re.MULTILINE),
        (b"xabc!", b"(?<=x)abc(?=!)", 0),
        (b"x abc! xabc", rb"\babc\b", 0),
    ),
)
def test_regex_chunks_preserve_full_buffer_context(
    data: bytes, pattern: bytes, flags: int
) -> None:
    expected = tuple(
        (match.start(), match.end(), match.group())
        for match in re.finditer(pattern, data, flags)
    )
    observed = RegexScanEngine(data, pattern, halo=0, flags=flags).run(chunk_size=2)
    observed_values = tuple((item.start, item.end, item.value) for item in observed)
    assert observed_values == expected
