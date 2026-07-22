"""Focused contracts for the private payload-aware transaction experiment."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.performance.experiments import rangeset_transaction as experiment
from treemendous import (
    JoinPayloadPolicy,
    OrderedPayloadPolicy,
    Span,
    UniformPayloadPolicy,
    create_range_set,
)
from treemendous.backends.adapters import BackendAdapter
from treemendous.backends.registry import BackendRegistry
from treemendous.backends.types import Available, Capability, Maturity
from treemendous.basic.boundary import IntervalManager
from treemendous.domain import MutationResult
from treemendous.rangeset import RangeSet


def _raw(ranges: RangeSet) -> tuple[tuple[int, int], ...]:
    return tuple((item.start, item.end) for item in ranges._adapter.intervals())


def _assert_unchanged(
    ranges: RangeSet, before_snapshot: Any, before_raw: tuple[tuple[int, int], ...]
) -> None:
    assert ranges.snapshot() == before_snapshot
    assert _raw(ranges) == before_raw


def _policy(kind: str) -> Any:
    if kind == "uniform":
        return UniformPayloadPolicy()
    if kind == "join":
        return JoinPayloadPolicy(lambda left, right: left | right, frozenset())
    if kind == "ordered":
        return OrderedPayloadPolicy(
            lambda left, right: left + right,
            (),
            event_key_fn=lambda value: value,
        )
    return None


def _payload(kind: str, label: int) -> Any:
    if kind == "uniform":
        return label
    if kind == "join":
        return frozenset({label})
    if kind == "ordered":
        return (label,)
    return experiment._MISSING


def _source(kind: str = "none", backend: str = "py_boundary") -> RangeSet:
    ranges = create_range_set(
        (0, 40),
        backend=backend,
        initially_available=False,
        payload_policy=_policy(kind),
    )
    payload = _payload(kind, 1)
    if payload is experiment._MISSING:
        ranges.add(Span(4, 12))
        ranges.add(Span(20, 28))
    else:
        ranges.add(Span(4, 12), payload)
        ranges.add(Span(20, 28), payload)
    return ranges


def _trace(kind: str) -> tuple[experiment._TransactionOperation, ...]:
    add_payload = _payload(kind, 1)
    return (
        experiment._TransactionOperation("discard", Span(6, 10)),
        experiment._TransactionOperation("add", Span(8, 16), add_payload),
        experiment._TransactionOperation("strict-discard", Span(0, 2)),
        experiment._TransactionOperation("discard", Span(22, 25)),
    )


@pytest.mark.parametrize("kind", ("none", "uniform", "join", "ordered"))
def test_transaction_matches_ordered_scalar_results_and_snapshot(kind: str) -> None:
    transaction_ranges = _source(kind)
    scalar_ranges = _source(kind)
    operations = _trace(kind)

    expected = experiment._scalar(scalar_ranges, operations)
    actual = experiment._rangeset_transaction(transaction_ranges, operations)

    assert actual == expected
    assert all(type(row) is MutationResult for row in actual)
    assert transaction_ranges.snapshot() == scalar_ranges.snapshot()
    assert actual[2] == MutationResult((), 0, False)


def test_empty_transaction_and_cancellation_non_restorative_traces() -> None:
    empty = _source()
    before = empty.snapshot()
    assert experiment._rangeset_transaction(empty, ()) == ()
    assert empty.snapshot() == before

    restorative = _source()
    scalar = _source()
    restore_trace = (
        experiment._TransactionOperation("discard", Span(4, 8)),
        experiment._TransactionOperation("add", Span(4, 8)),
    )
    assert experiment._rangeset_transaction(
        restorative, restore_trace
    ) == experiment._scalar(scalar, restore_trace)
    assert restorative.snapshot() == scalar.snapshot()

    application = _source()
    scalar_application = _source()
    non_restore = (
        experiment._TransactionOperation("add", Span(12, 14)),
        experiment._TransactionOperation("discard", Span(21, 24)),
        experiment._TransactionOperation("add", Span(30, 32)),
    )
    assert experiment._rangeset_transaction(
        application, non_restore
    ) == experiment._scalar(scalar_application, non_restore)
    assert application.snapshot() == scalar_application.snapshot()


def test_mutable_payloads_are_isolated_and_custom_cloner_is_used() -> None:
    clone_calls: list[list[str]] = []

    def cloner(value: list[str] | None) -> list[str] | None:
        clone_calls.append([] if value is None else list(value))
        return copy.deepcopy(value)

    ranges = create_range_set(
        (0, 20),
        backend="py_boundary",
        initially_available=False,
        payload_policy=UniformPayloadPolicy[list[str]](),
        payload_cloner=cloner,
    )
    initial = ["initial"]
    ranges.add(Span(2, 6), initial)
    incoming = ["next"]
    operations = (experiment._TransactionOperation("add", Span(8, 10), incoming),)
    experiment._rangeset_transaction(ranges, operations)
    incoming.append("mutated")
    observed = ranges.intervals()
    assert observed[1].data == ["next"]
    observed[1].data.append("escaped")
    assert ranges.intervals()[1].data == ["next"]
    assert len(clone_calls) > 4


def test_materialization_domain_and_result_failures_do_not_publish() -> None:
    ranges = _source()
    before = ranges.snapshot()
    raw = _raw(ranges)

    def broken_iterable() -> Iterator[Any]:
        yield ("discard", (4, 5))
        raise LookupError("iterator failed")

    with pytest.raises(LookupError, match="iterator failed"):
        experiment._rangeset_transaction(ranges, broken_iterable())
    _assert_unchanged(ranges, before, raw)

    with pytest.raises(ValueError, match="managed domain"):
        experiment._rangeset_transaction(ranges, (("add", (50, 51)),))
    _assert_unchanged(ranges, before, raw)

    def result_failure(*_args: Any) -> Any:
        raise RuntimeError("result failed")

    with pytest.raises(RuntimeError, match="result failed"):
        experiment._rangeset_transaction(
            ranges,
            (("discard", (4, 5)),),
            _result_factory=result_failure,
        )
    _assert_unchanged(ranges, before, raw)

    with pytest.raises(TypeError, match="exact MutationResult"):
        experiment._rangeset_transaction(
            ranges,
            (("discard", (4, 5)),),
            _result_factory=lambda *_args: object(),
        )
    _assert_unchanged(ranges, before, raw)


@pytest.mark.parametrize("failure_at", (1, 2, 3, 4, 5))
def test_injected_backend_failure_after_each_staged_prefix(
    monkeypatch: pytest.MonkeyPatch, failure_at: int
) -> None:
    ranges = _source("uniform")
    before = ranges.snapshot()
    raw = _raw(ranges)
    original_implementation = ranges._adapter.implementation
    calls = 0
    original_release = IntervalManager.release_interval
    original_reserve = IntervalManager.reserve_interval

    def checkpoint(instance: IntervalManager) -> None:
        nonlocal calls
        if instance is original_implementation:
            return
        calls += 1
        if calls == failure_at:
            raise RuntimeError(f"backend prefix {failure_at}")

    def release(instance: IntervalManager, start: int, end: int) -> None:
        checkpoint(instance)
        original_release(instance, start, end)

    def reserve(instance: IntervalManager, start: int, end: int) -> None:
        checkpoint(instance)
        original_reserve(instance, start, end)

    monkeypatch.setattr(IntervalManager, "release_interval", release)
    monkeypatch.setattr(IntervalManager, "reserve_interval", reserve)
    operations = (
        experiment._TransactionOperation("discard", Span(4, 6)),
        experiment._TransactionOperation("add", Span(12, 14), 1),
        experiment._TransactionOperation("discard", Span(20, 22)),
    )
    with pytest.raises(RuntimeError, match="backend prefix"):
        experiment._rangeset_transaction(ranges, operations)
    _assert_unchanged(ranges, before, raw)


def test_cloner_failure_does_not_publish() -> None:
    armed = False

    def cloner(value: Any) -> Any:
        if armed:
            raise RuntimeError("clone failed")
        return copy.deepcopy(value)

    ranges = create_range_set(
        (0, 10),
        backend="py_boundary",
        initially_available=False,
        payload_policy=UniformPayloadPolicy(),
        payload_cloner=cloner,
    )
    ranges.add(Span(1, 3), "x")
    before = ranges.snapshot()
    raw = _raw(ranges)
    armed = True
    with pytest.raises(RuntimeError, match="clone failed"):
        experiment._rangeset_transaction(ranges, ())
    armed = False
    _assert_unchanged(ranges, before, raw)


@pytest.mark.parametrize("callback", ("can_merge", "combine", "restrict", "cloner"))
def test_payload_callbacks_cannot_reenter_source(callback: str) -> None:
    armed = False
    source: RangeSet

    class Policy:
        def attempt(self, name: str) -> None:
            if armed and callback == name:
                source.add(Span(14, 15), "reentry")

        def can_merge(self, left: str, right: str) -> bool:
            self.attempt("can_merge")
            return left == right

        def combine(self, left: str, right: str) -> str:
            self.attempt("combine")
            return left

        def restrict(self, data: str, source_span: Span, target: Span) -> str:
            self.attempt("restrict")
            return data

    policy = Policy()

    def cloner(value: Any) -> Any:
        if armed and callback == "cloner":
            source.add(Span(14, 15), "reentry")
        return copy.deepcopy(value)

    source = create_range_set(
        (0, 20),
        backend="py_boundary",
        initially_available=False,
        payload_policy=policy,
        payload_cloner=cloner,
    )
    source.add(Span(2, 6), "x")
    before = source.snapshot()
    raw = _raw(source)
    armed = True
    operation = experiment._TransactionOperation("add", Span(4, 8), "x")
    with pytest.raises(RuntimeError, match="payload processing"):
        experiment._rangeset_transaction(source, (operation,))
    armed = False
    _assert_unchanged(source, before, raw)


def test_operation_iterator_cannot_reenter_source() -> None:
    source = _source()
    before = source.snapshot()
    raw = _raw(source)

    def operations() -> Iterator[Any]:
        source.discard(Span(4, 5))
        yield ("add", (12, 14))

    with pytest.raises(RuntimeError, match="reentrant mutation"):
        experiment._rangeset_transaction(source, operations())
    _assert_unchanged(source, before, raw)


def test_custom_adapter_is_rejected_before_operation_iteration() -> None:
    class DirectManager(IntervalManager):
        pass

    ranges = RangeSet(BackendAdapter(DirectManager()), domain=(0, 10))
    iterated = False

    def operations() -> Iterator[Any]:
        nonlocal iterated
        iterated = True
        yield ("discard", (0, 1))

    with pytest.raises(experiment._UnsupportedAdapterError, match="custom adapters"):
        experiment._rangeset_transaction(ranges, operations())
    assert not iterated


def test_all_discovered_stable_deterministic_core_factories_match_scalar() -> None:
    registry = BackendRegistry.discover()
    exercised: list[str] = []
    for spec in registry.specs:
        state = registry.states[spec.id]
        if not (
            isinstance(state, Available)
            and spec.maturity is Maturity.STABLE
            and spec.deterministic
            and Capability.CORE in spec.capabilities
        ):
            continue
        transaction_ranges = _source(backend=spec.id)
        scalar_ranges = _source(backend=spec.id)
        operations = _trace("none")
        actual = experiment._rangeset_transaction(transaction_ranges, operations)
        expected = experiment._scalar(scalar_ranges, operations)
        assert actual == expected, spec.id
        assert transaction_ranges.snapshot() == scalar_ranges.snapshot(), spec.id
        exercised.append(spec.id)
    assert exercised


@pytest.fixture(scope="module")
def focused_report() -> dict[str, Any]:
    return experiment.run_matrix(
        samples=15,
        interval_counts=(64,),
        batch_sizes=(0, 1, 4),
    )


def test_bounded_report_writes_and_verifies_strict_triplet(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output = tmp_path / "transaction.json"
    paths = experiment.write_artifacts(focused_report, output)
    verified = experiment.verify_artifacts(output)
    assert verified["gate"]["decision"] == "REJECTED"
    assert all(path.is_file() for path in paths)


def _rewrite(output: Path, text: str) -> None:
    encoded = text.encode()
    output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    Path(f"{output}.sha256").write_text(f"{digest}  {output.name}\n")


def test_verifier_rejects_duplicate_nonfinite_and_exact_type_tamper(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output = tmp_path / "transaction.json"
    experiment.write_artifacts(focused_report, output)
    duplicate = output.read_text().replace(
        f'  "schema": "{experiment.SCHEMA}"',
        f'  "schema": "duplicate",\n  "schema": "{experiment.SCHEMA}"',
        1,
    )
    _rewrite(output, duplicate)
    with pytest.raises(ValueError, match="duplicate key"):
        experiment.verify_artifacts(output)

    tampered = copy.deepcopy(focused_report)
    tampered["rows"][0]["paired_ratios"][0] = float("nan")
    text = json.dumps(tampered, indent=2, sort_keys=True) + "\n"
    _rewrite(output, text)
    with pytest.raises(ValueError, match="non-finite"):
        experiment.verify_artifacts(output)

    tampered = copy.deepcopy(focused_report)
    tampered["rows"][0]["scalar_ns_samples"][0] = True
    experiment.write_artifacts(tampered, output)
    with pytest.raises(ValueError, match="raw paired ratios|exact type"):
        experiment.verify_artifacts(output)


@pytest.mark.parametrize(
    ("expected_batch", "replacement"),
    ((0, False), (1, True)),
)
def test_verifier_rejects_false_zero_and_true_one_row_identity_tamper(
    tmp_path: Path,
    focused_report: dict[str, Any],
    expected_batch: int,
    replacement: bool,
) -> None:
    output = tmp_path / "transaction.json"
    tampered = copy.deepcopy(focused_report)
    row = next(row for row in tampered["rows"] if row["batch_size"] == expected_batch)
    row["batch_size"] = replacement
    experiment.write_artifacts(tampered, output)

    with pytest.raises(ValueError, match="row identity exact type"):
        experiment.verify_artifacts(output)


@pytest.mark.parametrize(
    ("index", "replacement"),
    ((0, False), (1, True)),
)
def test_verifier_rejects_false_zero_and_true_one_matrix_dimension_tamper(
    tmp_path: Path,
    focused_report: dict[str, Any],
    index: int,
    replacement: bool,
) -> None:
    output = tmp_path / "transaction.json"
    tampered = copy.deepcopy(focused_report)
    tampered["matrix"]["batch_sizes"][index] = replacement
    experiment.write_artifacts(tampered, output)

    with pytest.raises(ValueError, match="fixed matrix membership"):
        experiment.verify_artifacts(output)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("interval_count", False),
        ("backend", 0),
        ("payload", 0),
        ("trace", 0),
    ),
)
def test_verifier_exact_type_checks_every_row_identity_field(
    tmp_path: Path,
    focused_report: dict[str, Any],
    field: str,
    replacement: bool | int,
) -> None:
    output = tmp_path / "transaction.json"
    tampered = copy.deepcopy(focused_report)
    tampered["rows"][0][field] = replacement
    experiment.write_artifacts(tampered, output)

    with pytest.raises(ValueError, match="row identity exact type"):
        experiment.verify_artifacts(output)


@pytest.mark.parametrize(
    "path",
    (
        ("runtime", "python_version"),
        ("runtime", "python_implementation"),
        ("runtime", "python_compiler"),
        ("runtime", "platform"),
        ("runtime", "machine"),
        ("runtime", "architecture"),
        ("build", "command"),
        ("build", "cxx"),
        ("build", "cxx_version"),
        ("build", "cc"),
        ("build", "cflags"),
        ("build", "flags", "TREE_MENDOUS_WITH_ICL"),
        ("backend", "id"),
        ("backend", "module"),
        ("backend", "type"),
        ("backend", "path"),
        ("backend", "sha256"),
    ),
)
def test_verifier_rejects_each_runtime_build_and_backend_provenance_category(
    tmp_path: Path, focused_report: dict[str, Any], path: tuple[str, ...]
) -> None:
    output = tmp_path / "transaction.json"
    tampered = copy.deepcopy(focused_report)
    target = tampered["provenance"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = True
    experiment.write_artifacts(tampered, output)

    with pytest.raises(ValueError, match="runtime|build|active backend"):
        experiment.verify_artifacts(output)


def test_verifier_rejects_matrix_row_digest_method_and_provenance_tamper(
    tmp_path: Path, focused_report: dict[str, Any]
) -> None:
    output = tmp_path / "transaction.json"

    matrix = copy.deepcopy(focused_report)
    matrix["matrix"]["batch_sizes"] = [0, 1]
    experiment.write_artifacts(matrix, output)
    with pytest.raises(ValueError, match="fixed matrix membership"):
        experiment.verify_artifacts(output)

    reordered = copy.deepcopy(focused_report)
    reordered["rows"][0], reordered["rows"][1] = (
        reordered["rows"][1],
        reordered["rows"][0],
    )
    reordered["gate"] = experiment._gate(reordered["rows"])
    experiment.write_artifacts(reordered, output)
    with pytest.raises(ValueError, match="membership/order"):
        experiment.verify_artifacts(output)

    extra = copy.deepcopy(focused_report)
    extra["rows"][0]["extra"] = True
    experiment.write_artifacts(extra, output)
    with pytest.raises(ValueError, match="row keys/type"):
        experiment.verify_artifacts(output)

    report_extra = copy.deepcopy(focused_report)
    report_extra["extra"] = True
    experiment.write_artifacts(report_extra, output)
    with pytest.raises(ValueError, match="report keys/type"):
        experiment.verify_artifacts(output)

    gate = copy.deepcopy(focused_report)
    gate["gate"]["decision"] = "ACCEPTED"
    experiment.write_artifacts(gate, output)
    with pytest.raises(ValueError, match="gate mismatch"):
        experiment.verify_artifacts(output)

    result_digest = copy.deepcopy(focused_report)
    result_digest["rows"][0]["result_sha256"] = "0" * 64
    experiment.write_artifacts(result_digest, output)
    with pytest.raises(ValueError, match="result digest"):
        experiment.verify_artifacts(output)

    final_digest = copy.deepcopy(focused_report)
    final_digest["rows"][0]["final_state_sha256"] = "0" * 64
    experiment.write_artifacts(final_digest, output)
    with pytest.raises(ValueError, match="final-state digest"):
        experiment.verify_artifacts(output)

    method = copy.deepcopy(focused_report)
    method["matrix"]["timed_boundary"] = "tampered"
    experiment.write_artifacts(method, output)
    with pytest.raises(ValueError, match="methodology"):
        experiment.verify_artifacts(output)

    source = copy.deepcopy(focused_report)
    source["provenance"]["sources"][0]["sha256"] = "0" * 64
    experiment.write_artifacts(source, output)
    with pytest.raises(ValueError, match="source provenance"):
        experiment.verify_artifacts(output)

    binary = copy.deepcopy(focused_report)
    binary["provenance"]["backend"]["sha256"] = "0" * 64
    experiment.write_artifacts(binary, output)
    with pytest.raises(ValueError, match="active backend"):
        experiment.verify_artifacts(output)
