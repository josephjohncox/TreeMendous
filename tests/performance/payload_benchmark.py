"""Backend-differential load qualification for explicit payload policies."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

from treemendous import (
    JoinPayloadPolicy,
    OrderedPayloadPolicy,
    PayloadPolicy,
    Span,
    UniformPayloadPolicy,
    create_range_set,
)


@dataclass(frozen=True)
class PayloadOperation:
    kind: str
    start: int | None = None
    end: int | None = None
    payload: Any = None
    length: int = 1
    not_before: int = 0
    predicate: str | None = None


@dataclass(frozen=True)
class PayloadScenario:
    name: str
    plausible_use: str
    domain: tuple[tuple[int, int], ...]
    initially_available: bool
    policy_factory: Callable[[], PayloadPolicy[Any]]
    setup: tuple[PayloadOperation, ...]
    operations: tuple[PayloadOperation, ...]


def _uniform_policy() -> PayloadPolicy[Any]:
    return UniformPayloadPolicy()


def _join_policy() -> PayloadPolicy[Any]:
    return JoinPayloadPolicy(join=lambda left, right: left | right, bottom=frozenset())


def _ordered_policy() -> PayloadPolicy[Any]:
    return OrderedPayloadPolicy(
        combine_fn=lambda left, right: left + right,
        identity=(),
        event_key_fn=lambda value: value,
    )


def _predicate(mode: str | None) -> Callable[[Any], bool] | None:
    if mode is None:
        return None
    if mode.startswith("equal:"):
        expected = mode.removeprefix("equal:")
        return lambda value: value == expected
    if mode.startswith("contains:"):
        expected = mode.removeprefix("contains:")
        return lambda value: expected in value
    if mode == "nonempty":
        return bool
    raise ValueError(f"unknown payload predicate: {mode}")


def _canonical(value: Any) -> Any:
    if isinstance(value, frozenset):
        return {"frozenset": sorted(_canonical(item) for item in value)}
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"payload benchmark value is not canonical: {type(value).__name__}")


def _digest(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _new_target(backend_id: str, scenario: PayloadScenario):
    return create_range_set(
        scenario.domain,
        backend=backend_id,
        initially_available=scenario.initially_available,
        payload_policy=scenario.policy_factory(),
    )


def _replay(target: Any, operations: tuple[PayloadOperation, ...], *, observe: bool):
    queries: list[Any] = []
    for operation in operations:
        predicate = _predicate(operation.predicate)
        result: Any = None
        if operation.kind == "add":
            assert operation.start is not None and operation.end is not None
            target.add(Span(operation.start, operation.end), operation.payload)
        elif operation.kind == "discard":
            assert operation.start is not None and operation.end is not None
            target.discard(Span(operation.start, operation.end))
        elif operation.kind in {"first_fit", "allocate"}:
            result = getattr(target, operation.kind)(
                operation.length,
                not_before=operation.not_before,
                payload_predicate=predicate,
            )
        elif operation.kind == "overlaps":
            assert operation.start is not None and operation.end is not None
            result = target.overlaps(Span(operation.start, operation.end))
        elif operation.kind == "snapshot":
            result = target.snapshot()
        elif operation.kind == "stats":
            result = target.stats()
        else:
            raise ValueError(f"unknown payload operation: {operation.kind}")
        if observe and operation.kind in {
            "first_fit",
            "allocate",
            "overlaps",
            "snapshot",
            "stats",
        }:
            observed: Any
            if operation.kind in {"first_fit", "allocate"}:
                observed = (
                    None
                    if result is None
                    else (result.start, result.end, _canonical(result.data))
                )
            elif operation.kind == "overlaps":
                observed = tuple(
                    (item.start, item.end, _canonical(item.data)) for item in result
                )
            elif operation.kind == "snapshot":
                observed = (
                    result.total_free,
                    tuple(
                        (item.start, item.end, _canonical(item.data))
                        for item in result.intervals
                    ),
                )
            else:
                observed = (
                    result.total_free,
                    result.total_occupied,
                    result.free_chunks,
                    result.largest_chunk,
                    result.bounds,
                )
            queries.append((operation.kind, observed))
    if not observe:
        return None
    state = tuple(
        (item.start, item.end, _canonical(item.data)) for item in target.intervals()
    )
    snapshot = target.snapshot()
    return {
        "state": state,
        "queries": tuple(queries),
        "total_free": snapshot.total_free,
    }


def _uniform_scenario(scale: int, operations: int, seed: int) -> PayloadScenario:
    extent = scale * 4
    setup = tuple(
        PayloadOperation(
            "add", start=index * 4, end=index * 4 + 2, payload=f"tenant-{index % 8}"
        )
        for index in range(scale)
    )
    rng = random.Random(seed)
    trace: list[PayloadOperation] = []
    for index in range(operations):
        segment = rng.randrange(scale)
        start = segment * 4
        label = f"tenant-{segment % 8}"
        selector = index % 7
        if selector == 0:
            trace.append(PayloadOperation("discard", start=start, end=start + 1))
        elif selector == 1:
            trace.append(
                PayloadOperation("add", start=start, end=start + 2, payload=label)
            )
        elif selector == 2:
            trace.append(
                PayloadOperation(
                    "first_fit",
                    not_before=start,
                    predicate=f"equal:{label}",
                )
            )
        elif selector == 3:
            trace.append(
                PayloadOperation(
                    "allocate",
                    not_before=start,
                    predicate=f"equal:{label}",
                )
            )
        elif selector == 4:
            trace.append(PayloadOperation("overlaps", start=start, end=start + 2))
        elif selector == 5:
            trace.append(PayloadOperation("snapshot"))
        else:
            trace.append(PayloadOperation("stats"))
    return PayloadScenario(
        "uniform-tenant-capacity",
        "tenant-labelled capacity with exact-label allocation",
        ((0, extent),),
        False,
        _uniform_policy,
        setup,
        tuple(trace),
    )


def _overlay_scenario(
    *, ordered: bool, scale: int, operations: int, seed: int
) -> PayloadScenario:
    extent = max(256, scale * 8)
    rng = random.Random(seed)
    setup: list[PayloadOperation] = []
    for index in range(scale):
        start = rng.randrange(0, extent - 16)
        end = min(extent, start + rng.randint(2, 32))
        payload: Any = (
            (f"event-{index % 16}",) if ordered else frozenset({f"team-{index % 16}"})
        )
        setup.append(PayloadOperation("add", start=start, end=end, payload=payload))
    trace: list[PayloadOperation] = []
    for index in range(operations):
        start = rng.randrange(0, extent - 16)
        end = min(extent, start + rng.randint(2, 24))
        selector = index % 7
        label = f"event-{index % 16}" if ordered else f"team-{index % 16}"
        payload = (label,) if ordered else frozenset({label})
        predicate = "nonempty" if ordered else f"contains:{label}"
        if selector == 0:
            trace.append(PayloadOperation("add", start=start, end=end, payload=payload))
        elif selector == 1:
            trace.append(PayloadOperation("discard", start=start, end=end))
        elif selector == 2:
            trace.append(
                PayloadOperation("first_fit", not_before=start, predicate=predicate)
            )
        elif selector == 3:
            trace.append(
                PayloadOperation("allocate", not_before=start, predicate=predicate)
            )
        elif selector == 4:
            trace.append(PayloadOperation("overlaps", start=start, end=end))
        elif selector == 5:
            trace.append(PayloadOperation("snapshot"))
        else:
            trace.append(PayloadOperation("stats"))
    return PayloadScenario(
        "ordered-booking-events" if ordered else "joined-access-overlays",
        (
            "ordered booking/event overlays"
            if ordered
            else "commutative access or ownership overlays"
        ),
        ((0, extent),),
        True,
        _ordered_policy if ordered else _join_policy,
        tuple(setup),
        tuple(trace),
    )


def qualify_payload_backends(
    backend_ids: tuple[str, ...], *, scale: int, operations: int, seed: int = 42
) -> list[dict[str, Any]]:
    """Exercise every payload policy and reject cross-backend divergence."""
    if min(scale, operations) <= 0:
        raise ValueError("payload scale and operations must be positive")
    scenarios = (
        _uniform_scenario(scale, operations, seed),
        _overlay_scenario(
            ordered=False, scale=scale, operations=operations, seed=seed + 1
        ),
        _overlay_scenario(
            ordered=True, scale=scale, operations=operations, seed=seed + 2
        ),
    )
    reports: list[dict[str, Any]] = []
    for scenario in scenarios:
        expected: Any = None
        results: dict[str, Any] = {}
        for backend_id in backend_ids:
            validation_target = _new_target(backend_id, scenario)
            _replay(validation_target, scenario.setup, observe=False)
            observed = _replay(validation_target, scenario.operations, observe=True)
            if expected is None:
                expected = observed
            elif observed != expected:
                raise AssertionError(
                    f"{backend_id} diverged in payload scenario {scenario.name}"
                )

            timed_target = _new_target(backend_id, scenario)
            setup_started = time.perf_counter_ns()
            _replay(timed_target, scenario.setup, observe=False)
            setup_ns = time.perf_counter_ns() - setup_started
            execution_started = time.perf_counter_ns()
            _replay(timed_target, scenario.operations, observe=False)
            execution_ns = time.perf_counter_ns() - execution_started
            timed_observed = _replay(timed_target, (), observe=True)
            if timed_observed["state"] != expected["state"]:
                raise AssertionError(
                    f"{backend_id} timed payload state diverged in {scenario.name}"
                )
            results[backend_id] = {
                "setup_ns": setup_ns,
                "execution_ns": execution_ns,
                "operations_per_second": operations * 1_000_000_000 / execution_ns,
            }
        reports.append(
            {
                "label": (
                    "cross-backend payload-policy qualification; common semantics "
                    "are independently covered by payload law tests"
                ),
                "workload": scenario.name,
                "plausible_use": scenario.plausible_use,
                "dataset": {
                    "setup_operations": len(scenario.setup),
                    "timed_operations": len(scenario.operations),
                },
                "validation": {
                    "state_checksum": _digest(expected["state"]),
                    "query_checksum": _digest(expected["queries"]),
                    "final_intervals": len(expected["state"]),
                    "query_observations": len(expected["queries"]),
                    "total_free": expected["total_free"],
                },
                "results": results,
            }
        )
    return reports
