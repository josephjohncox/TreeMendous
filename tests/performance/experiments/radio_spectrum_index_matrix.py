#!/usr/bin/env python3
"""Radio-spectrum BoxIndex injection and adaptive-representation experiment.

This module is deliberately experiment-only.  It replaces ``scheduler._index``
only while the newly constructed scheduler is empty; no production factory,
constructor, decision policy, or live migration path is added.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import math
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from treemendous.applications.scheduling.radio_spectrum import (
    RadioSpectrumScheduler,
    SpectrumConflictError,
    SpectrumReservation,
    SpectrumStatus,
)
from treemendous.multidimensional import BoundedBoxIndex, Box, BoxIndex, BoxIndex2D

SCHEMA = "treemendous-radio-spectrum-index-matrix-v2"
POLICY_VERSION = "radio-spectrum-index-policy-v2"
MINIMUM_BLOCKS = 25
TRAINING_SIZES = (32, 128, 512, 2_000, 10_000)
HELD_OUT_SIZES = (64, 256, 1_000, 5_000)
CHANNEL_COUNT = 64
OPERATIONS_PER_POSITION = 8
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 7_316_219
PROJECTION_ELIGIBILITY_LIMIT = 0.80
SELECTED_CELL_LIMIT = 0.90
HELD_OUT_SELECTED_LIMIT = 0.90
DEFAULT_LINEAR_LIMIT = 1.10
DEFAULT_LINEAR_CONTROL_SIZE = 512
DEFAULT_LINEAR_CONTROL_SEED = 527_911
MEMORY_LIMIT = 1.25
GRID_LIMITS = {
    "max_total_cells": 20_000,
    "max_cells_per_entry": 32,
    "max_cells_per_query": 20_000,
    "max_total_postings": 200_000,
    "max_estimated_bytes": 128 * 1024 * 1024,
}
CANDIDATES = ("projection", "grid")
BUILD_FLAG_NAMES = (
    "BOOST_ROOT",
    "TREE_MENDOUS_DISABLE_OPTIMIZATIONS",
    "TREE_MENDOUS_GLIBCXX_DEBUG",
    "TREE_MENDOUS_LOCAL_NATIVE",
    "TREE_MENDOUS_SANITIZERS",
    "TREE_MENDOUS_WITH_ICL",
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_PATHS = (
    "tests/performance/experiments/radio_spectrum_index_matrix.py",
    "treemendous/applications/scheduling/radio_spectrum.py",
    "treemendous/multidimensional/index.py",
    "treemendous/multidimensional/diagnostics.py",
    "treemendous/multidimensional/algorithms/projection.py",
    "treemendous/multidimensional/algorithms/grid.py",
)


@dataclasses.dataclass(frozen=True)
class Scenario:
    """One fixed query shape, density, mutation mix, and skew cell."""

    name: str
    density: Literal["low", "medium", "high"]
    channel_query: Literal["narrow", "broad"]
    time_query: Literal["narrow", "broad"]
    insertion_cancellation_ratio: Literal["3:1", "2:2", "1:3", "replay"]
    axis_skew: Literal["balanced", "channel", "time", "both"]


SCENARIOS = (
    Scenario("low-narrow-narrow", "low", "narrow", "narrow", "3:1", "balanced"),
    Scenario("medium-broad-channel", "medium", "broad", "narrow", "2:2", "channel"),
    Scenario("medium-broad-time", "medium", "narrow", "broad", "2:2", "time"),
    Scenario("high-broad-both", "high", "broad", "broad", "1:3", "both"),
    Scenario(
        "idempotent-replay",
        "medium",
        "narrow",
        "narrow",
        "replay",
        "balanced",
    ),
)


@dataclasses.dataclass(frozen=True)
class _OracleRecord:
    reservation: SpectrumReservation
    box: Box


class _RadioOracle:
    """Independent sorted-list radio conflict oracle (no BoxIndex dependency)."""

    def __init__(self, channel_count: int) -> None:
        self.channel_count = channel_count
        self.records: dict[str, _OracleRecord] = {}
        self.next_by_owner: dict[str, int] = {}
        self.requests: dict[tuple[str, str], tuple[tuple[int, ...], str]] = {}
        self.version = 0

    def _box(
        self, channel_start: int, channel_width: int, start: int, end: int, guard: int
    ) -> Box:
        if channel_width < 1 or channel_start + channel_width > self.channel_count:
            raise ValueError("requested channels exceed the managed channel domain")
        return Box(
            (max(0, channel_start - guard), start),
            (min(self.channel_count, channel_start + channel_width + guard), end),
        )

    def overlaps(self, box: Box) -> tuple[str, ...]:
        return tuple(
            sorted(
                record.reservation.id
                for record in self.records.values()
                if record.reservation.active and record.box.overlaps(box)
            )
        )

    def reserve(
        self,
        owner: str,
        channel_start: int,
        channel_width: int,
        start: int,
        end: int,
        *,
        guard_channels: int = 0,
        request_id: str | None = None,
    ) -> SpectrumReservation:
        fingerprint = (channel_start, channel_width, start, end, guard_channels)
        if request_id is not None:
            prior = self.requests.get((owner, request_id))
            if prior is not None:
                if prior[0] != fingerprint:
                    raise ValueError(
                        "idempotency key was already used for a different request"
                    )
                return self.records[prior[1]].reservation
        box = self._box(channel_start, channel_width, start, end, guard_channels)
        conflicts = self.overlaps(box)
        if conflicts:
            raise SpectrumConflictError(
                # Use the production value carrier, but not its index or algorithm.
                __import__(
                    "treemendous.applications.scheduling.radio_spectrum",
                    fromlist=["SpectrumConflict"],
                ).SpectrumConflict(box, conflicts)
            )
        sequence = self.next_by_owner.get(owner, 1)
        reservation_id = f"{owner}:{sequence}"
        reservation = SpectrumReservation(
            reservation_id,
            owner,
            channel_start,
            channel_start + channel_width,
            start,
            end,
            guard_channels,
            request_id,
        )
        self.records[reservation_id] = _OracleRecord(reservation, box)
        self.next_by_owner[owner] = sequence + 1
        if request_id is not None:
            self.requests[(owner, request_id)] = fingerprint, reservation_id
        self.version += 1
        return reservation

    def conflicts_for(
        self,
        channel_start: int,
        channel_width: int,
        start: int,
        end: int,
        *,
        guard_channels: int = 0,
    ) -> tuple[Box, tuple[str, ...]] | None:
        box = self._box(channel_start, channel_width, start, end, guard_channels)
        conflicts = self.overlaps(box)
        return None if not conflicts else (box, conflicts)

    def cancel(self, owner: str, reservation_id: str) -> SpectrumReservation:
        record = self.records.get(reservation_id)
        if record is None:
            raise KeyError(reservation_id)
        if record.reservation.owner != owner:
            raise PermissionError("reservation belongs to a different owner")
        if not record.reservation.active:
            return record.reservation
        cancelled = dataclasses.replace(
            record.reservation, status=SpectrumStatus.CANCELLED
        )
        self.records[reservation_id] = _OracleRecord(cancelled, record.box)
        self.version += 1
        return cancelled


def _time_upper(active_entries: int, blocks: int, scenario_count: int) -> int:
    # The most channel-skewed seed uses 16 channel positions with an eight-unit
    # low-density stride. Mutation reservations occupy disjoint later windows.
    seed_upper = ((active_entries + 15) // 16) * 8
    return max(1_024, seed_upper + blocks * scenario_count * 8 + 64)


def _new_index(
    kind: str, *, active_entries: int, blocks: int, scenario_count: int
) -> Any:
    if kind == "linear":
        return BoxIndex(2)
    if kind == "projection":
        return BoxIndex2D()
    if kind == "grid":
        upper = _time_upper(active_entries, blocks, scenario_count)
        return BoundedBoxIndex(
            Box((0, 0), (CHANNEL_COUNT, upper)),
            (4, 8),
            max_total_cells=GRID_LIMITS["max_total_cells"],
            max_cells_per_entry=GRID_LIMITS["max_cells_per_entry"],
            max_cells_per_query=GRID_LIMITS["max_cells_per_query"],
            max_total_postings=GRID_LIMITS["max_total_postings"],
            max_estimated_bytes=GRID_LIMITS["max_estimated_bytes"],
        )
    raise ValueError(f"unknown experiment index: {kind}")


def _new_scheduler(
    kind: str, *, active_entries: int, blocks: int, scenario_count: int
) -> RadioSpectrumScheduler:
    """Inject one index into an empty real scheduler, with no fallback path."""
    scheduler = RadioSpectrumScheduler(CHANNEL_COUNT)
    if scheduler._records or len(scheduler._index) or scheduler._index.dimensions != 2:
        raise AssertionError("experiment injection requires a new empty scheduler")
    scheduler._index = _new_index(
        kind,
        active_entries=active_entries,
        blocks=blocks,
        scenario_count=scenario_count,
    )
    return scheduler


def _seed_profile(scenario: Scenario) -> tuple[tuple[int, ...], int, int]:
    channels = {
        "balanced": tuple(range(0, CHANNEL_COUNT, 2)),
        "channel": tuple(range(24, 40)),
        "time": tuple(range(CHANNEL_COUNT)),
        "both": tuple(range(24, 40)),
    }[scenario.axis_skew]
    duration, stride = {
        "low": (1, 8),
        "medium": (2, 4),
        "high": (3, 3),
    }[scenario.density]
    return channels, duration, stride


def _seed_arguments(scenario: Scenario, index: int) -> dict[str, Any]:
    channels, duration, stride = _seed_profile(scenario)
    start = (index // len(channels)) * stride
    return {
        "owner": f"seed-{index}",
        "channel_start": channels[index % len(channels)],
        "channel_width": 1,
        "start": start,
        "end": start + duration,
        "guard_channels": 0,
        "request_id": f"seed-request-{index}",
    }


def _seed_oracle(scenario: Scenario, active_entries: int) -> _RadioOracle:
    oracle = _RadioOracle(CHANNEL_COUNT)
    for index in range(active_entries):
        oracle.reserve(**_seed_arguments(scenario, index))
    return oracle


def _seed_scheduler(
    scheduler: RadioSpectrumScheduler, active_entries: int, scenario: Scenario
) -> _RadioOracle:
    oracle = _RadioOracle(CHANNEL_COUNT)
    for index in range(active_entries):
        arguments = _seed_arguments(scenario, index)
        actual = scheduler.reserve(**arguments)
        expected = oracle.reserve(**arguments)
        if actual != expected:
            raise AssertionError("seed reservation diverged from independent oracle")
    _assert_state(scheduler, oracle)
    return oracle


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _deep_size(root: Any) -> int:
    """Deterministic CPython retained-graph proxy with identity de-duplication."""
    seen: set[int] = set()

    def visit(value: Any) -> int:
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        total = sys.getsizeof(value)
        if isinstance(value, dict):
            return total + sum(visit(key) + visit(item) for key, item in value.items())
        if isinstance(value, (tuple, list, set, frozenset)):
            return total + sum(visit(item) for item in value)
        namespace = getattr(value, "__dict__", None)
        if isinstance(namespace, dict):
            total += visit(namespace)
        slots = getattr(type(value), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if hasattr(value, slot):
                total += visit(getattr(value, slot))
        return total

    return visit(root)


def _construct(
    kind: str,
    *,
    active_entries: int,
    blocks: int,
    scenario_count: int,
    scenario: Scenario,
) -> tuple[RadioSpectrumScheduler, _RadioOracle, dict[str, int | str]]:
    rss_before = _rss_bytes()
    started = time.perf_counter_ns()
    scheduler = _new_scheduler(
        kind,
        active_entries=active_entries,
        blocks=blocks,
        scenario_count=scenario_count,
    )
    oracle = _seed_scheduler(scheduler, active_entries, scenario)
    construction_ns = time.perf_counter_ns() - started
    rss_after = _rss_bytes()
    return (
        scheduler,
        oracle,
        {
            "algorithm": scheduler._index.diagnostics().algorithm,
            "construction_ns": construction_ns,
            "retained_bytes": _deep_size(scheduler._index),
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "rss_high_water_delta_bytes": max(0, rss_after - rss_before),
        },
    )


def _reservation(value: SpectrumReservation) -> dict[str, Any]:
    return {
        "id": value.id,
        "owner": value.owner,
        "channel_start": value.channel_start,
        "channel_end": value.channel_end,
        "start": value.start,
        "end": value.end,
        "guard_channels": value.guard_channels,
        "request_id": value.request_id,
        "status": value.status.value,
    }


def _state(scheduler: RadioSpectrumScheduler) -> dict[str, Any]:
    snapshot = scheduler.snapshot()
    diagnostics = scheduler._index.diagnostics()
    return {
        "reservations": [_reservation(item) for item in snapshot.reservations],
        "geometry": {
            "dimensions": snapshot.geometry.dimensions,
            "version": snapshot.geometry.version,
            "entries": [
                {
                    "sequence": entry.handle.sequence,
                    "lower": list(entry.box.lower),
                    "upper": list(entry.box.upper),
                    "data": entry.data,
                }
                for entry in snapshot.geometry.entries
            ],
        },
        "diagnostics": {
            "algorithm": diagnostics.algorithm,
            "dimensions": diagnostics.dimensions,
            "version": diagnostics.version,
            "entry_count": diagnostics.entry_count,
            "distinct_box_count": diagnostics.distinct_box_count,
            "duplicate_entry_count": diagnostics.duplicate_entry_count,
            "projection_sizes": list(diagnostics.projection_sizes),
            "posting_count": diagnostics.posting_count,
            "occupied_cell_count": diagnostics.occupied_cell_count,
            "estimated_memory_bytes": diagnostics.estimated_memory_bytes,
        },
    }


def _oracle_state(oracle: _RadioOracle, algorithm: str) -> dict[str, Any]:
    reservations = sorted(
        (record.reservation for record in oracle.records.values()),
        key=lambda item: (item.start, item.channel_start, item.id),
    )
    active = [
        record
        for record in oracle.records.values()
        if record.reservation.status is SpectrumStatus.ACTIVE
    ]
    # Handles are allocated monotonically only for successful insertions.
    entries = []
    sequence = 0
    for record in oracle.records.values():
        if record.reservation.active:
            # Record iteration order includes cancelled history; recover the handle
            # sequence from successful-insert order by counting all records.
            pass
    active_ids = {record.reservation.id for record in active}
    for sequence, record in enumerate(oracle.records.values(), 1):
        if record.reservation.id in active_ids:
            entries.append(
                {
                    "sequence": sequence,
                    "lower": list(record.box.lower),
                    "upper": list(record.box.upper),
                    "data": record.reservation.id,
                }
            )
    boxes = [record.box for record in active]
    distinct = len(set(boxes))
    strategy = {
        "algorithm": algorithm,
        "dimensions": 2,
        "version": oracle.version,
        "entry_count": len(active),
        "distinct_box_count": distinct,
        "duplicate_entry_count": len(active) - distinct,
        "projection_sizes": [len(active), len(active)]
        if algorithm == "axis_projection"
        else [],
        "posting_count": None,
        "occupied_cell_count": None,
        "estimated_memory_bytes": None,
    }
    return {
        "reservations": [_reservation(item) for item in reservations],
        "geometry": {"dimensions": 2, "version": oracle.version, "entries": entries},
        "diagnostics": strategy,
    }


def _assert_state(scheduler: RadioSpectrumScheduler, oracle: _RadioOracle) -> None:
    actual = _state(scheduler)
    expected = _oracle_state(oracle, actual["diagnostics"]["algorithm"])
    # Grid-only retained diagnostics are exact structural evidence, not oracle data.
    if actual["diagnostics"]["algorithm"] == "sparse_grid":
        expected["diagnostics"].update(
            {
                "posting_count": actual["diagnostics"]["posting_count"],
                "occupied_cell_count": actual["diagnostics"]["occupied_cell_count"],
                "estimated_memory_bytes": actual["diagnostics"][
                    "estimated_memory_bytes"
                ],
            }
        )
    if actual != expected:
        raise AssertionError("real scheduler state diverged from independent oracle")


def _observed(call: Callable[[], Any]) -> dict[str, Any]:
    try:
        value = call()
        if isinstance(value, SpectrumReservation):
            result: Any = _reservation(value)
        elif value is None:
            result = None
        else:
            result = {
                "requested": [list(value.requested.lower), list(value.requested.upper)],
                "conflicting_ids": list(value.conflicting_ids),
            }
        return {"result": result}
    except Exception as error:  # noqa: BLE001 - exact exception is experiment evidence
        detail: dict[str, Any] = {
            "module": type(error).__module__,
            "type": type(error).__qualname__,
            "message": str(error),
        }
        if isinstance(error, SpectrumConflictError):
            detail["conflicting_ids"] = list(error.conflict.conflicting_ids)
        return {"exception": detail}


def _semantic_trace(kind: str) -> dict[str, Any]:
    scheduler = _new_scheduler(
        kind, active_entries=4, blocks=MINIMUM_BLOCKS, scenario_count=4
    )
    oracle = _RadioOracle(CHANNEL_COUNT)
    arguments: dict[str, Any] = {
        "owner": "alpha",
        "channel_start": 5,
        "channel_width": 2,
        "start": 10,
        "end": 20,
        "guard_channels": 1,
        "request_id": "packet",
    }
    first = scheduler.reserve(**arguments)
    expected_first = oracle.reserve(**arguments)
    replay = scheduler.reserve(**arguments)
    expected_replay = oracle.reserve(**arguments)
    calls = [
        {"name": "reserve", **_observed(lambda: first)},
        {
            "name": "idempotent-replay",
            "same_identity": replay is first,
            **_observed(lambda: replay),
        },
        {
            "name": "idempotency-mismatch",
            **_observed(lambda: scheduler.reserve(**{**arguments, "end": 21})),
        },
        {
            "name": "conflicting-reserve",
            **_observed(
                lambda: scheduler.reserve("beta", 8, 1, 12, 14, guard_channels=1)
            ),
        },
        {
            "name": "conflicts-for",
            **_observed(lambda: scheduler.conflicts_for(5, 2, 12, 14)),
        },
    ]
    oracle_mismatch = _observed(lambda: oracle.reserve(**{**arguments, "end": 21}))
    oracle_conflict = _observed(
        lambda: oracle.reserve("beta", 8, 1, 12, 14, guard_channels=1)
    )
    oracle_query_raw = oracle.conflicts_for(5, 2, 12, 14)
    if oracle_query_raw is None:
        raise AssertionError("oracle query unexpectedly had no conflict")
    touching = scheduler.reserve("gamma", 5, 2, 20, 25, guard_channels=1)
    oracle_touching = oracle.reserve("gamma", 5, 2, 20, 25, guard_channels=1)
    cancelled = scheduler.cancel("alpha", first.id)
    oracle_cancelled = oracle.cancel("alpha", expected_first.id)
    cancelled_again = scheduler.cancel("alpha", first.id)
    oracle_cancelled_again = oracle.cancel("alpha", expected_first.id)
    replacement = scheduler.reserve("delta", 5, 2, 10, 20, guard_channels=1)
    oracle_replacement = oracle.reserve("delta", 5, 2, 10, 20, guard_channels=1)
    calls.extend(
        [
            {"name": "touching-time", **_observed(lambda: touching)},
            {"name": "cancel", **_observed(lambda: cancelled)},
            {
                "name": "cancel-idempotent",
                "same_identity": cancelled_again is cancelled,
                **_observed(lambda: cancelled_again),
            },
            {"name": "historical-duplicate-geometry", **_observed(lambda: replacement)},
            {
                "name": "wrong-owner",
                **_observed(lambda: scheduler.cancel("wrong", replacement.id)),
            },
            {
                "name": "missing",
                **_observed(lambda: scheduler.cancel("delta", "missing")),
            },
        ]
    )
    expected_calls = [
        {"name": "reserve", **_observed(lambda: expected_first)},
        {
            "name": "idempotent-replay",
            "same_identity": expected_replay is expected_first,
            **_observed(lambda: expected_replay),
        },
        {"name": "idempotency-mismatch", **oracle_mismatch},
        {"name": "conflicting-reserve", **oracle_conflict},
        {
            "name": "conflicts-for",
            "result": {
                "requested": [
                    list(oracle_query_raw[0].lower),
                    list(oracle_query_raw[0].upper),
                ],
                "conflicting_ids": list(oracle_query_raw[1]),
            },
        },
        {"name": "touching-time", **_observed(lambda: oracle_touching)},
        {"name": "cancel", **_observed(lambda: oracle_cancelled)},
        {
            "name": "cancel-idempotent",
            "same_identity": oracle_cancelled_again is oracle_cancelled,
            **_observed(lambda: oracle_cancelled_again),
        },
        {
            "name": "historical-duplicate-geometry",
            **_observed(lambda: oracle_replacement),
        },
        {
            "name": "wrong-owner",
            **_observed(lambda: oracle.cancel("wrong", oracle_replacement.id)),
        },
        {
            "name": "missing",
            **_observed(lambda: oracle.cancel("delta", "missing")),
        },
    ]
    if calls != expected_calls:
        raise AssertionError("scheduler calls diverged from independent radio oracle")
    _assert_state(scheduler, oracle)
    state = _state(scheduler)
    return {
        "kind": kind,
        "algorithm": state["diagnostics"]["algorithm"],
        "calls": calls,
        "snapshot_order": [item["id"] for item in state["reservations"]],
        "snapshot_version": state["geometry"]["version"],
        "final_state": state,
        "final_state_sha256": _sha(state),
    }


def _grid_guard_adversary() -> dict[str, Any]:
    scheduler = RadioSpectrumScheduler(CHANNEL_COUNT)
    setattr(
        scheduler,
        "_index",
        BoundedBoxIndex(
            Box((0, 0), (CHANNEL_COUNT, 1_024)),
            (4, 8),
            max_total_cells=2_048,
            max_cells_per_entry=4,
            max_cells_per_query=64,
            max_total_postings=128,
            max_estimated_bytes=2 * 1024 * 1024,
        ),
    )
    scheduler.reserve("guard", 0, 1, 0, 2)
    before = _state(scheduler)
    try:
        scheduler.conflicts_for(0, CHANNEL_COUNT, 0, 1_024)
    except ValueError as error:
        observed = {
            "module": type(error).__module__,
            "type": type(error).__qualname__,
            "message": str(error),
        }
    else:
        raise AssertionError("grid guard error was swallowed or fell back")
    after = _state(scheduler)
    if before != after or "max_cells_per_query" not in observed["message"]:
        raise AssertionError("grid guard did not propagate atomically")
    return {
        "algorithm": scheduler._index.diagnostics().algorithm,
        "bounds": [[0, 0], [CHANNEL_COUNT, 1_024]],
        "cell_size": [4, 8],
        "limits": {
            "max_total_cells": 2_048,
            "max_cells_per_entry": 4,
            "max_cells_per_query": 64,
            "max_total_postings": 128,
            "max_estimated_bytes": 2 * 1024 * 1024,
        },
        "error": observed,
        "state_unchanged": True,
        "fallback": None,
    }


def _query_arguments(
    scenario: Scenario, active_entries: int, query_index: int
) -> tuple[int, int, int, int]:
    seed = _seed_arguments(scenario, query_index % active_entries)
    channel_start = int(seed["channel_start"])
    channel_width = int(seed["channel_width"])
    start = int(seed["start"])
    end = int(seed["end"])
    padding = {"low": 0, "medium": 1, "high": 2}[scenario.density]
    if padding:
        channel_start = max(0, channel_start - padding)
        channel_end = min(CHANNEL_COUNT, channel_start + channel_width + 2 * padding)
        channel_width = channel_end - channel_start
        start = max(0, start - padding)
        end += padding
    if scenario.channel_query == "broad":
        channel_start, channel_width = 0, CHANNEL_COUNT
    if scenario.time_query == "broad":
        last = _seed_arguments(scenario, active_entries - 1)
        start, end = 0, int(last["end"])
    return channel_start, channel_width, start, end


_Command = tuple[str, tuple[Any, ...], dict[str, Any]]


def _mutation_arguments(
    scenario: Scenario,
    *,
    active_entries: int,
    scenario_index: int,
    block_index: int,
    total_blocks: int,
    seed: int,
    slot: int,
) -> tuple[tuple[Any, ...], dict[str, Any], str]:
    last_seed = _seed_arguments(scenario, active_entries - 1)
    start = (
        int(last_seed["end"]) + 8 + (scenario_index * total_blocks + block_index) * 8
    )
    owner = f"timed-{seed}-{scenario.name}-{block_index}-{slot}"
    args = (owner, (block_index * 7 + slot) % CHANNEL_COUNT, 1, start, start + 2)
    kwargs = {"request_id": f"request-{block_index}-{slot}"}
    return args, kwargs, f"{owner}:1"


def _commands(
    scenario: Scenario,
    *,
    active_entries: int,
    scenario_index: int,
    block_index: int,
    total_blocks: int,
    seed: int,
) -> tuple[tuple[_Command, ...], tuple[_Command, ...]]:
    rng = random.Random(seed + scenario_index * 100_003 + block_index)
    timed: list[_Command] = []
    setup: list[_Command] = []
    for _ in range(4):
        args = _query_arguments(scenario, active_entries, rng.randrange(active_entries))
        timed.append(("conflicts_for", args, {}))

    mutations = [
        _mutation_arguments(
            scenario,
            active_entries=active_entries,
            scenario_index=scenario_index,
            block_index=block_index,
            total_blocks=total_blocks,
            seed=seed,
            slot=slot,
        )
        for slot in range(4)
    ]
    ratio = scenario.insertion_cancellation_ratio
    if ratio == "3:1":
        timed.extend(("reserve", args, kwargs) for args, kwargs, _ in mutations[:3])
        timed.append(("cancel", (mutations[0][0][0], mutations[0][2]), {}))
    elif ratio == "2:2":
        timed.extend(("reserve", args, kwargs) for args, kwargs, _ in mutations[:2])
        timed.extend(
            ("cancel", (args[0], reservation_id), {})
            for args, _, reservation_id in mutations[:2]
        )
    elif ratio == "1:3":
        setup.extend(("reserve", args, kwargs) for args, kwargs, _ in mutations[:3])
        timed.append(("reserve", mutations[3][0], mutations[3][1]))
        timed.extend(
            ("cancel", (args[0], reservation_id), {})
            for args, _, reservation_id in mutations[:3]
        )
    else:
        setup.append(("reserve", mutations[0][0], mutations[0][1]))
        timed.extend(("reserve", mutations[0][0], mutations[0][1]) for _ in range(4))

    if len(timed) != OPERATIONS_PER_POSITION:
        raise AssertionError("timed position operation count changed")
    return tuple(setup), tuple(timed)


def _execute(target: Any, commands: Sequence[_Command]) -> tuple[Any, ...]:
    results: list[Any] = []
    for method, args, kwargs in commands:
        value = getattr(target, method)(*args, **kwargs)
        if method == "conflicts_for":
            if value is None:
                results.append(None)
            elif isinstance(value, tuple):
                results.append((value[0], tuple(value[1])))
            else:
                results.append((value.requested, value.conflicting_ids))
        else:
            results.append(value)
    return tuple(results)


def _normalize_results(values: Sequence[Any]) -> list[Any]:
    normalized: list[Any] = []
    for value in values:
        if value is None:
            normalized.append(None)
        elif isinstance(value, SpectrumReservation):
            normalized.append(_reservation(value))
        else:
            normalized.append(
                {
                    "requested": [list(value[0].lower), list(value[0].upper)],
                    "ids": list(value[1]),
                }
            )
    return normalized


def _candidate_count(index: Any, box: Box) -> int:
    with index._lock:
        state = index._state
        return len(
            index._strategy.candidate_handles(state.strategy_state, box, state.entries)
        )


def _query_diagnostics(
    scheduler: RadioSpectrumScheduler,
    scenario: Scenario,
    active_entries: int,
    seed: int,
) -> dict[str, Any]:
    candidates = []
    matches = []
    for query in range(16):
        args = _query_arguments(
            scenario,
            active_entries,
            random.Random(seed + query).randrange(active_entries),
        )
        box = Box((args[0], args[2]), (args[0] + args[1], args[3]))
        candidates.append(_candidate_count(scheduler._index, box))
        matches.append(len(scheduler._index.overlaps(box)))
    diagnostics = scheduler._index.diagnostics()
    return {
        "samples": len(candidates),
        "candidate_counts": candidates,
        "candidate_median": float(statistics.median(candidates)),
        "candidate_max": max(candidates),
        "match_counts": matches,
        "match_median": float(statistics.median(matches)),
        "match_max": max(matches),
        "posting_count": diagnostics.posting_count,
        "algorithm": diagnostics.algorithm,
    }


def _bootstrap(values: Sequence[float], seed: int) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize empty samples")
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return {
        "median": float(statistics.median(values)),
        "median_95_low": float(medians[int(0.025 * len(medians))]),
        "median_95_high": float(medians[int(0.975 * len(medians))]),
    }


def _normalized_state(state: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(state, allow_nan=False))
    result["diagnostics"].update(
        {
            "algorithm": "normalized",
            "projection_sizes": [],
            "posting_count": None,
            "occupied_cell_count": None,
            "estimated_memory_bytes": None,
        }
    )
    return result


def _measure_scenario(
    candidate_scheduler: RadioSpectrumScheduler,
    baseline_scheduler: RadioSpectrumScheduler,
    candidate_oracle: _RadioOracle,
    baseline_oracle: _RadioOracle,
    *,
    phase: str,
    candidate: str,
    scenario: Scenario,
    scenario_index: int,
    active_entries: int,
    blocks: int,
    seed: int,
) -> dict[str, Any]:
    raw_blocks = []
    ratios = []
    candidate_ns_values: list[int] = []
    baseline_ns_values: list[int] = []
    for block_index in range(blocks):
        setup, commands = _commands(
            scenario,
            active_entries=active_entries,
            scenario_index=scenario_index,
            block_index=block_index,
            total_blocks=blocks,
            seed=seed,
        )
        setup_expected = _normalize_results(_execute(candidate_oracle, setup))
        if _normalize_results(_execute(baseline_oracle, setup)) != setup_expected:
            raise AssertionError("independent oracle setup instances diverged")
        if (
            _normalize_results(_execute(candidate_scheduler, setup)) != setup_expected
            or _normalize_results(_execute(baseline_scheduler, setup)) != setup_expected
        ):
            raise AssertionError("untimed mutation setup diverged from oracle")
        oracle_results = _execute(candidate_oracle, commands)
        other_oracle_results = _execute(baseline_oracle, commands)
        if _normalize_results(oracle_results) != _normalize_results(
            other_oracle_results
        ):
            raise AssertionError("independent oracle instances diverged")
        order = (
            ("candidate", candidate_scheduler, candidate_ns_values)
            if block_index % 2 == 0
            else ("baseline", baseline_scheduler, baseline_ns_values)
        )
        second = (
            ("baseline", baseline_scheduler, baseline_ns_values)
            if block_index % 2 == 0
            else ("candidate", candidate_scheduler, candidate_ns_values)
        )
        durations: dict[str, int] = {}
        results: dict[str, tuple[Any, ...]] = {}
        for label, scheduler, sink in (order, second):
            started = time.perf_counter_ns()
            result = _execute(scheduler, commands)
            duration = time.perf_counter_ns() - started
            durations[label] = duration
            results[label] = result
            sink.append(duration)
        expected = _normalize_results(oracle_results)
        if (
            _normalize_results(results["candidate"]) != expected
            or _normalize_results(results["baseline"]) != expected
        ):
            raise AssertionError("timed scheduler result diverged from oracle")
        # Exact same timed instances are validated only after both timers stop.
        _assert_state(candidate_scheduler, candidate_oracle)
        _assert_state(baseline_scheduler, baseline_oracle)
        candidate_state = _normalized_state(_state(candidate_scheduler))
        baseline_state = _normalized_state(_state(baseline_scheduler))
        if candidate_state != baseline_state:
            raise AssertionError("candidate and linear scheduler final states diverged")
        ratio = durations["candidate"] / durations["baseline"]
        ratios.append(ratio)
        raw_blocks.append(
            {
                "block": block_index,
                "order": [order[0], second[0]],
                "operations": len(commands),
                "candidate_ns": durations["candidate"],
                "baseline_ns": durations["baseline"],
                "ratio": ratio,
            }
        )
    digest_state = _normalized_state(_state(candidate_scheduler))
    return {
        "phase": phase,
        "candidate": candidate,
        "scenario": dataclasses.asdict(scenario),
        "active_entries": active_entries,
        "seed": seed,
        "blocks": raw_blocks,
        "ratios": ratios,
        "ratio": _bootstrap(ratios, BOOTSTRAP_SEED + seed + active_entries),
        "candidate_latency_ns": _bootstrap(
            [float(value) for value in candidate_ns_values],
            BOOTSTRAP_SEED + seed + active_entries + 1,
        ),
        "baseline_latency_ns": _bootstrap(
            [float(value) for value in baseline_ns_values],
            BOOTSTRAP_SEED + seed + active_entries + 2,
        ),
        "operations_per_position": OPERATIONS_PER_POSITION,
        "validated_blocks": blocks,
        "query_diagnostics": _query_diagnostics(
            candidate_scheduler, scenario, active_entries, seed
        ),
        "final_state_sha256": _sha(digest_state),
    }


def _measure_phase(
    *,
    phase: str,
    sizes: Sequence[int],
    scenarios: Sequence[Scenario],
    candidates: Sequence[str],
    blocks: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    construction = []
    rows = []
    for candidate in candidates:
        for active_entries in sizes:
            for scenario_index, scenario in enumerate(scenarios):
                baseline, baseline_oracle, baseline_metrics = _construct(
                    "linear",
                    active_entries=active_entries,
                    blocks=blocks,
                    scenario_count=len(scenarios),
                    scenario=scenario,
                )
                tested, tested_oracle, tested_metrics = _construct(
                    candidate,
                    active_entries=active_entries,
                    blocks=blocks,
                    scenario_count=len(scenarios),
                    scenario=scenario,
                )
                retained_ratio = int(tested_metrics["retained_bytes"]) / int(
                    baseline_metrics["retained_bytes"]
                )
                construction.append(
                    {
                        "phase": phase,
                        "candidate": candidate,
                        "scenario": dataclasses.asdict(scenario),
                        "active_entries": active_entries,
                        "baseline": baseline_metrics,
                        "candidate_metrics": tested_metrics,
                        "retained_memory_ratio": retained_ratio,
                    }
                )
                rows.append(
                    _measure_scenario(
                        tested,
                        baseline,
                        tested_oracle,
                        baseline_oracle,
                        phase=phase,
                        candidate=candidate,
                        scenario=scenario,
                        scenario_index=scenario_index,
                        active_entries=active_entries,
                        blocks=blocks,
                        seed=seed,
                    )
                )
    return construction, rows


def _measure_default_linear_control(blocks: int) -> dict[str, Any]:
    """Measure the untouched constructor against an explicitly injected linear index."""
    scenario = SCENARIOS[0]
    default = RadioSpectrumScheduler(CHANNEL_COUNT)
    default_oracle = _seed_scheduler(default, DEFAULT_LINEAR_CONTROL_SIZE, scenario)
    control = _new_scheduler(
        "linear",
        active_entries=DEFAULT_LINEAR_CONTROL_SIZE,
        blocks=blocks,
        scenario_count=len(SCENARIOS),
    )
    control_oracle = _seed_scheduler(control, DEFAULT_LINEAR_CONTROL_SIZE, scenario)
    return _measure_scenario(
        default,
        control,
        default_oracle,
        control_oracle,
        phase="default_linear_control",
        candidate="default",
        scenario=scenario,
        scenario_index=0,
        active_entries=DEFAULT_LINEAR_CONTROL_SIZE,
        blocks=blocks,
        seed=DEFAULT_LINEAR_CONTROL_SEED,
    )


@dataclasses.dataclass(frozen=True)
class DecisionEvidence:
    """Immutable experiment decision; it is not a runtime selection object."""

    policy_version: str
    decision: str
    runtime_index: str
    runtime_seam_retained: bool
    live_migration: bool
    selected_cells: tuple[str, ...]
    rejected_reasons: tuple[str, ...]
    default_linear_upper_ratio: float


def _cell_id(row: dict[str, Any]) -> str:
    return f"{row['candidate']}:{row['scenario']['name']}:{row['active_entries']}"


def _gate(
    training: Sequence[dict[str, Any]],
    held_out: Sequence[dict[str, Any]],
    construction: Sequence[dict[str, Any]],
    training_sizes: Sequence[int],
    default_linear_control: dict[str, Any],
) -> dict[str, Any]:
    memory = {
        (
            row["phase"],
            row["candidate"],
            row["scenario"]["name"],
            row["active_entries"],
        ): row["retained_memory_ratio"]
        for row in construction
    }
    eligible: dict[tuple[str, str], int] = {}
    reasons = []
    for candidate in sorted({row["candidate"] for row in training}):
        for scenario in sorted({row["scenario"]["name"] for row in training}):
            by_size = {
                row["active_entries"]: row
                for row in training
                if row["candidate"] == candidate and row["scenario"]["name"] == scenario
            }
            crossover = None
            for left, right in zip(training_sizes, training_sizes[1:], strict=False):
                if (
                    left in by_size
                    and right in by_size
                    and by_size[left]["ratio"]["median_95_high"]
                    <= PROJECTION_ELIGIBILITY_LIMIT
                    and by_size[right]["ratio"]["median_95_high"]
                    <= PROJECTION_ELIGIBILITY_LIMIT
                ):
                    # The crossover is the lower member of the first passing
                    # adjacent pair, not the first size after that boundary.
                    crossover = left
                    break
            if crossover is None:
                reasons.append(
                    f"{candidate}/{scenario}: no adjacent training sizes have upper-95 <= {PROJECTION_ELIGIBILITY_LIMIT:.2f}"
                )
            else:
                eligible[(candidate, scenario)] = crossover
    selected_training = []
    for row in training:
        key = (row["candidate"], row["scenario"]["name"])
        if (
            key in eligible
            and row["active_entries"] >= eligible[key]
            and row["ratio"]["median_95_high"] <= SELECTED_CELL_LIMIT
            and memory[
                (
                    "training",
                    row["candidate"],
                    row["scenario"]["name"],
                    row["active_entries"],
                )
            ]
            <= MEMORY_LIMIT
        ):
            selected_training.append(row)
    selected_held_out = []
    held_out_failures = []
    missing_held_out = []
    selected_keys = {
        (row["candidate"], row["scenario"]["name"]) for row in selected_training
    }
    for key in sorted(selected_keys):
        evidence = [
            row
            for row in held_out
            if (row["candidate"], row["scenario"]["name"]) == key
            and row["active_entries"] >= eligible[key]
        ]
        if not evidence:
            missing_held_out.append(f"{key[0]}/{key[1]}@>={eligible[key]}")
            continue
        for row in evidence:
            memory_ok = (
                memory[
                    (
                        "held_out",
                        row["candidate"],
                        row["scenario"]["name"],
                        row["active_entries"],
                    )
                ]
                <= MEMORY_LIMIT
            )
            latency_ok = row["ratio"]["median_95_high"] <= HELD_OUT_SELECTED_LIMIT
            if memory_ok and latency_ok:
                selected_held_out.append(row)
            else:
                held_out_failures.append(_cell_id(row))
    if eligible and not selected_training:
        reasons.append(
            "eligible latency cells all failed the 1.25x retained-memory gate"
        )
    if missing_held_out:
        reasons.append(
            "selected crossovers lack predetermined held-out evidence: "
            + ", ".join(missing_held_out)
        )
    if held_out_failures:
        reasons.append(
            "held-out selected cells failed latency or memory: "
            + ", ".join(sorted(held_out_failures))
        )
    default_linear_upper = default_linear_control["ratio"]["median_95_high"]
    if default_linear_upper > DEFAULT_LINEAR_LIMIT:
        reasons.append(
            "measured default scheduler exceeded explicitly patched linear control: "
            f"upper-95 {default_linear_upper:.4f} > {DEFAULT_LINEAR_LIMIT:.2f}"
        )
    selected = tuple(
        sorted(_cell_id(row) for row in [*selected_training, *selected_held_out])
    )
    qualifies = (
        bool(selected_training)
        and not held_out_failures
        and not missing_held_out
        and default_linear_upper <= DEFAULT_LINEAR_LIMIT
    )
    if not qualifies and not reasons:
        reasons.append("no stable crossover passed all fixed gates")
    decision = DecisionEvidence(
        policy_version=POLICY_VERSION,
        decision="QUALIFIED_PRIVATE_CONFIRMATION_REQUIRED" if qualifies else "REJECTED",
        runtime_index="linear",
        runtime_seam_retained=False,
        live_migration=False,
        selected_cells=selected if qualifies else (),
        rejected_reasons=tuple(reasons),
        default_linear_upper_ratio=default_linear_upper,
    )
    material = dataclasses.asdict(decision)
    material["selected_cells"] = list(decision.selected_cells)
    material["rejected_reasons"] = list(decision.rejected_reasons)
    return material


def _sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        shown = str(resolved.relative_to(_REPOSITORY_ROOT))
    except ValueError:
        shown = str(resolved)
    return {"path": shown, "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}


def _git_metadata() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    changed = sorted(line[3:] for line in status.splitlines() if len(line) >= 4)
    staged = sorted(
        path for path in git("diff", "--cached", "--name-only").splitlines() if path
    )
    digest = hashlib.sha256(git("rev-parse", "HEAD").strip().encode())
    for relative in changed:
        path = _REPOSITORY_ROOT / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {
        "commit": git("rev-parse", "HEAD").strip(),
        "head_tree": git("rev-parse", "HEAD^{tree}").strip(),
        "clean_worktree": not bool(status),
        "changed_paths": changed,
        "staged_files": staged,
        "source_state_sha256": digest.hexdigest(),
    }


def _backend_provenance() -> list[dict[str, Any]]:
    result = []
    for name, implementation in (
        ("linear", BoxIndex),
        ("projection", BoxIndex2D),
        ("grid", BoundedBoxIndex),
    ):
        result.append(
            {
                "id": name,
                "module": implementation.__module__,
                "type": implementation.__qualname__,
                "source": _file(Path(inspect.getfile(implementation))),
            }
        )
    return result


def _provenance() -> dict[str, Any]:
    return {
        "git": _git_metadata(),
        "sources": [_file(_REPOSITORY_ROOT / path) for path in _SOURCE_PATHS],
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_compiler": platform.python_compiler(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "executable": sys.executable,
        },
        "build": {
            "command": os.environ.get(
                "TREE_MENDOUS_BUILD_COMMAND", "uv sync --all-extras"
            ),
            "flags": {name: os.environ.get(name, "") for name in BUILD_FLAG_NAMES},
        },
        "backends": _backend_provenance(),
    }


def _methodology(
    *,
    profile: str,
    blocks: int,
    training_sizes: Sequence[int],
    held_out_sizes: Sequence[int],
    scenarios: Sequence[Scenario],
    candidates: Sequence[str],
) -> dict[str, Any]:
    return {
        "profile": profile,
        "minimum_blocks": MINIMUM_BLOCKS,
        "blocks_recorded_before_run": blocks,
        "training_sizes": list(training_sizes),
        "held_out_sizes": list(held_out_sizes),
        "training_seed": 81_337,
        "held_out_seed": 914_221,
        "scenarios": [dataclasses.asdict(item) for item in scenarios],
        "candidates": list(candidates),
        "default_linear_control": {
            "active_entries": DEFAULT_LINEAR_CONTROL_SIZE,
            "scenario": SCENARIOS[0].name,
            "seed": DEFAULT_LINEAR_CONTROL_SEED,
            "comparison": "untouched constructor / explicitly injected BoxIndex(2)",
        },
        "operations_per_timed_position": OPERATIONS_PER_POSITION,
        "balanced_order": "fixed AB/BA blocks alternating candidate/linear and linear/candidate",
        "bootstrap_unit": "whole balanced block",
        "setup_timing": "scheduler/index construction, seeding, and distinct live cancellation-handle preparation excluded from operation latency",
        "validation_timing": "exact same timed instances validated against independent oracle after timing",
        "construction_timing": "real scheduler construction, private empty-index injection, and seed reservations",
        "memory_method": "recursive retained Python graph bytes plus process RSS high-water delta proxy",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "fixed_gates": {
            "adjacent_training_upper_95": PROJECTION_ELIGIBILITY_LIMIT,
            "selected_training_upper_95": SELECTED_CELL_LIMIT,
            "selected_held_out_upper_95": HELD_OUT_SELECTED_LIMIT,
            "default_linear_upper_95": DEFAULT_LINEAR_LIMIT,
            "selected_retained_memory_ratio": MEMORY_LIMIT,
        },
        "live_migration": False,
    }


def run_matrix(
    *,
    blocks: int = MINIMUM_BLOCKS,
    profile: str = "full",
    training_sizes: Sequence[int] | None = None,
    held_out_sizes: Sequence[int] | None = None,
    scenarios: Sequence[Scenario] | None = None,
    candidates: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the fixed full matrix or an explicitly marked focused test profile."""
    if type(blocks) is not int or blocks < MINIMUM_BLOCKS:
        raise ValueError(f"blocks must be an integer >= {MINIMUM_BLOCKS}")
    if profile not in {"full", "focused"}:
        raise ValueError("profile must be full or focused")
    training_sizes = tuple(training_sizes or TRAINING_SIZES)
    held_out_sizes = tuple(held_out_sizes or HELD_OUT_SIZES)
    scenarios = tuple(scenarios or SCENARIOS)
    candidates = tuple(candidates or CANDIDATES)
    if profile == "full" and (
        training_sizes != TRAINING_SIZES
        or held_out_sizes != HELD_OUT_SIZES
        or scenarios != SCENARIOS
        or candidates != CANDIDATES
    ):
        raise ValueError("full profile dimensions are fixed")
    if (
        not training_sizes
        or not held_out_sizes
        or not scenarios
        or len(set(training_sizes)) != len(training_sizes)
        or len(set(held_out_sizes)) != len(held_out_sizes)
        or len({item.name for item in scenarios}) != len(scenarios)
        or len(set(candidates)) != len(candidates)
        or any(candidate not in CANDIDATES for candidate in candidates)
        or any(item not in SCENARIOS for item in scenarios)
    ):
        raise ValueError("matrix dimensions must be nonempty fixed-domain values")
    methodology = _methodology(
        profile=profile,
        blocks=blocks,
        training_sizes=training_sizes,
        held_out_sizes=held_out_sizes,
        scenarios=scenarios,
        candidates=candidates,
    )
    correctness = [_semantic_trace(kind) for kind in ("linear", *candidates)]
    adversaries = {
        "broad_query": {
            "scenario": dataclasses.asdict(
                next(item for item in SCENARIOS if item.name == "high-broad-both")
            ),
            "required_in_training": any(
                item.name == "high-broad-both" for item in scenarios
            ),
        },
        "grid_guard": _grid_guard_adversary(),
    }
    training_construction, training = _measure_phase(
        phase="training",
        sizes=training_sizes,
        scenarios=scenarios,
        candidates=candidates,
        blocks=blocks,
        seed=methodology["training_seed"],
    )
    held_construction, held_out = _measure_phase(
        phase="held_out",
        sizes=held_out_sizes,
        scenarios=scenarios,
        candidates=candidates,
        blocks=blocks,
        seed=methodology["held_out_seed"],
    )
    construction = [*training_construction, *held_construction]
    default_linear_control = _measure_default_linear_control(blocks)
    decision = _gate(
        training,
        held_out,
        construction,
        training_sizes,
        default_linear_control,
    )
    return {
        "schema": SCHEMA,
        "methodology": methodology,
        "correctness": correctness,
        "adversaries": adversaries,
        "construction": construction,
        "training": training,
        "held_out": held_out,
        "default_linear_control": default_linear_control,
        "decision": decision,
        "provenance": _provenance(),
    }


def _markdown(report: dict[str, Any], digest: str) -> str:
    lines = [
        "# RadioSpectrumScheduler index representation experiment",
        "",
        f"Decision: **{report['decision']['decision']}**",
        f"Retained runtime index: **{report['decision']['runtime_index']}**",
        f"Runtime seam retained: **{report['decision']['runtime_seam_retained']}**",
        f"JSON SHA-256: `{digest}`",
        "",
        "| phase | candidate | scenario | active | median ratio | upper 95% | candidate median ns | memory ratio |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    memory = {
        (
            row["phase"],
            row["candidate"],
            row["scenario"]["name"],
            row["active_entries"],
        ): row["retained_memory_ratio"]
        for row in report["construction"]
    }
    for row in [*report["training"], *report["held_out"]]:
        lines.append(
            f"| {row['phase']} | {row['candidate']} | {row['scenario']['name']} | "
            f"{row['active_entries']} | {row['ratio']['median']:.4f} | "
            f"{row['ratio']['median_95_high']:.4f} | "
            f"{row['candidate_latency_ns']['median']:.0f} | "
            f"{memory[(row['phase'], row['candidate'], row['scenario']['name'], row['active_entries'])]:.4f} |"
        )
    lines.extend(("", "## Rejected reasons", ""))
    reasons = report["decision"]["rejected_reasons"]
    lines.extend(f"- {reason}" for reason in reasons)
    if not reasons:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_artifacts(report: dict[str, Any], output: Path) -> tuple[Path, Path, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    markdown = output.with_suffix(".md")
    markdown.write_text(_markdown(report, digest))
    checksum = Path(f"{output}.sha256")
    checksum.write_text(f"{digest}  {output.name}\n")
    return output, markdown, checksum


def _duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"non-finite number: {value}")


def _finite(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite number: {value}")
    return result


def _exact(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _exact(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _expected_row_evidence(
    candidate: str,
    scenario: Scenario,
    *,
    active_entries: int,
    scenario_index: int,
    scenario_count: int,
    blocks: int,
    seed: int,
) -> tuple[dict[str, Any], str]:
    scheduler = (
        RadioSpectrumScheduler(CHANNEL_COUNT)
        if candidate == "default"
        else _new_scheduler(
            candidate,
            active_entries=active_entries,
            blocks=blocks,
            scenario_count=scenario_count,
        )
    )
    _seed_scheduler(scheduler, active_entries, scenario)
    for block_index in range(blocks):
        setup, timed = _commands(
            scenario,
            active_entries=active_entries,
            scenario_index=scenario_index,
            block_index=block_index,
            total_blocks=blocks,
            seed=seed,
        )
        _execute(scheduler, setup)
        _execute(scheduler, timed)
    return (
        _query_diagnostics(scheduler, scenario, active_entries, seed),
        _sha(_normalized_state(_state(scheduler))),
    )


def _verify_row(
    row: Any,
    blocks: int,
    *,
    phase: str,
    candidate: str,
    scenario: Scenario,
    scenario_index: int,
    scenario_count: int,
    active_entries: int,
    seed_value: int,
) -> None:
    keys = {
        "phase",
        "candidate",
        "scenario",
        "active_entries",
        "seed",
        "blocks",
        "ratios",
        "ratio",
        "candidate_latency_ns",
        "baseline_latency_ns",
        "operations_per_position",
        "validated_blocks",
        "query_diagnostics",
        "final_state_sha256",
    }
    if type(row) is not dict or set(row) != keys:
        raise ValueError("benchmark row schema mismatch")
    if (
        row["phase"] != phase
        or type(row["phase"]) is not str
        or row["candidate"] != candidate
        or type(row["candidate"]) is not str
        or not _exact(row["scenario"], dataclasses.asdict(scenario))
        or type(row["active_entries"]) is not int
        or row["active_entries"] != active_entries
        or type(row["seed"]) is not int
        or row["seed"] != seed_value
        or type(row["validated_blocks"]) is not int
        or row["validated_blocks"] != blocks
        or type(row["operations_per_position"]) is not int
        or row["operations_per_position"] != OPERATIONS_PER_POSITION
    ):
        raise ValueError("benchmark row exact type/value mismatch")
    raw = row["blocks"]
    if type(raw) is not list or len(raw) != blocks:
        raise ValueError("raw block count mismatch")
    ratios = []
    candidate_ns = []
    baseline_ns = []
    for index, block in enumerate(raw):
        if type(block) is not dict or set(block) != {
            "block",
            "order",
            "operations",
            "candidate_ns",
            "baseline_ns",
            "ratio",
        }:
            raise ValueError("raw block schema mismatch")
        expected_order = (
            ["candidate", "baseline"] if index % 2 == 0 else ["baseline", "candidate"]
        )
        if (
            type(block["block"]) is not int
            or block["block"] != index
            or type(block["order"]) is not list
            or block["order"] != expected_order
            or type(block["operations"]) is not int
            or block["operations"] != OPERATIONS_PER_POSITION
            or type(block["candidate_ns"]) is not int
            or type(block["baseline_ns"]) is not int
            or min(block["candidate_ns"], block["baseline_ns"]) <= 0
            or type(block["ratio"]) is not float
        ):
            raise ValueError("raw block exact type/value mismatch")
        ratio = block["candidate_ns"] / block["baseline_ns"]
        if block["ratio"] != ratio:
            raise ValueError("raw ratio mismatch")
        ratios.append(ratio)
        candidate_ns.append(float(block["candidate_ns"]))
        baseline_ns.append(float(block["baseline_ns"]))
    bootstrap_seed = BOOTSTRAP_SEED + seed_value + active_entries
    if not _exact(row["ratios"], ratios) or not _exact(
        row["ratio"], _bootstrap(ratios, bootstrap_seed)
    ):
        raise ValueError("derived ratio interval mismatch")
    if not _exact(
        row["candidate_latency_ns"], _bootstrap(candidate_ns, bootstrap_seed + 1)
    ):
        raise ValueError("derived candidate latency mismatch")
    if not _exact(
        row["baseline_latency_ns"], _bootstrap(baseline_ns, bootstrap_seed + 2)
    ):
        raise ValueError("derived baseline latency mismatch")

    diagnostics = row["query_diagnostics"]
    diagnostic_keys = {
        "samples",
        "candidate_counts",
        "candidate_median",
        "candidate_max",
        "match_counts",
        "match_median",
        "match_max",
        "posting_count",
        "algorithm",
    }
    expected_algorithm = {
        "default": "linear",
        "projection": "axis_projection",
        "grid": "sparse_grid",
    }[candidate]
    if type(diagnostics) is not dict or set(diagnostics) != diagnostic_keys:
        raise ValueError("query diagnostics schema mismatch")
    candidate_counts = diagnostics["candidate_counts"]
    match_counts = diagnostics["match_counts"]
    if (
        type(diagnostics["samples"]) is not int
        or diagnostics["samples"] != 16
        or type(candidate_counts) is not list
        or type(match_counts) is not list
        or len(candidate_counts) != 16
        or len(match_counts) != 16
        or any(type(item) is not int or item < 0 for item in candidate_counts)
        or any(type(item) is not int or item < 0 for item in match_counts)
        or any(
            match > possible
            for match, possible in zip(match_counts, candidate_counts, strict=True)
        )
        or type(diagnostics["candidate_median"]) is not float
        or diagnostics["candidate_median"] != float(statistics.median(candidate_counts))
        or type(diagnostics["candidate_max"]) is not int
        or diagnostics["candidate_max"] != max(candidate_counts)
        or type(diagnostics["match_median"]) is not float
        or diagnostics["match_median"] != float(statistics.median(match_counts))
        or type(diagnostics["match_max"]) is not int
        or diagnostics["match_max"] != max(match_counts)
        or type(diagnostics["algorithm"]) is not str
        or diagnostics["algorithm"] != expected_algorithm
        or (candidate != "grid" and diagnostics["posting_count"] is not None)
        or (
            candidate == "grid"
            and (
                type(diagnostics["posting_count"]) is not int
                or diagnostics["posting_count"] < 0
            )
        )
    ):
        raise ValueError("query diagnostics exact/derived mismatch")

    expected_diagnostics, expected_digest = _expected_row_evidence(
        candidate,
        scenario,
        active_entries=active_entries,
        scenario_index=scenario_index,
        scenario_count=scenario_count,
        blocks=blocks,
        seed=seed_value,
    )
    if not _exact(diagnostics, expected_diagnostics):
        raise ValueError("reconstructed query diagnostics mismatch")
    if (
        type(row["final_state_sha256"]) is not str
        or row["final_state_sha256"] != expected_digest
    ):
        raise ValueError("final state digest mismatch")


def verify_artifacts(output: Path) -> dict[str, Any]:
    """Strictly parse, type-check, and recompute the canonical artifact triplet."""
    encoded = output.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if Path(f"{output}.sha256").read_text() != f"{digest}  {output.name}\n":
        raise ValueError("checksum mismatch")
    report = json.loads(
        encoded,
        object_pairs_hook=_duplicates,
        parse_constant=_nonfinite,
        parse_float=_finite,
    )
    top = {
        "schema",
        "methodology",
        "correctness",
        "adversaries",
        "construction",
        "training",
        "held_out",
        "default_linear_control",
        "decision",
        "provenance",
    }
    if type(report) is not dict or set(report) != top or report["schema"] != SCHEMA:
        raise ValueError("report schema mismatch")
    methodology = report["methodology"]
    if type(methodology) is not dict:
        raise ValueError("methodology schema mismatch")
    blocks = methodology.get("blocks_recorded_before_run")
    if type(blocks) is not int or blocks < MINIMUM_BLOCKS:
        raise ValueError("methodology block exact type/value mismatch")
    try:
        scenarios = tuple(Scenario(**item) for item in methodology["scenarios"])
        training_sizes = methodology["training_sizes"]
        held_out_sizes = methodology["held_out_sizes"]
        candidates = methodology["candidates"]
        expected_methodology = _methodology(
            profile=methodology["profile"],
            blocks=blocks,
            training_sizes=training_sizes,
            held_out_sizes=held_out_sizes,
            scenarios=scenarios,
            candidates=candidates,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("fixed methodology mismatch") from error
    if (
        not _exact(methodology, expected_methodology)
        or type(training_sizes) is not list
        or type(held_out_sizes) is not list
        or type(candidates) is not list
        or not training_sizes
        or not held_out_sizes
        or not scenarios
        or any(
            type(size) is not int or size <= 0
            for size in [*training_sizes, *held_out_sizes]
        )
        or len(set(training_sizes)) != len(training_sizes)
        or len(set(held_out_sizes)) != len(held_out_sizes)
        or len(set(candidates)) != len(candidates)
        or len({scenario.name for scenario in scenarios}) != len(scenarios)
        or any(candidate not in CANDIDATES for candidate in candidates)
        or any(scenario not in SCENARIOS for scenario in scenarios)
        or (
            methodology["profile"] == "full"
            and (
                tuple(training_sizes) != TRAINING_SIZES
                or tuple(held_out_sizes) != HELD_OUT_SIZES
                or scenarios != SCENARIOS
                or tuple(candidates) != CANDIDATES
            )
        )
        or methodology["profile"] not in {"full", "focused"}
    ):
        raise ValueError("fixed methodology mismatch")

    def matrix_specs(
        phase: str, sizes: Sequence[int], seed_value: int
    ) -> list[tuple[str, str, Scenario, int, int, int]]:
        return [
            (phase, candidate, scenario, scenario_index, active_entries, seed_value)
            for candidate in candidates
            for active_entries in sizes
            for scenario_index, scenario in enumerate(scenarios)
        ]

    training_specs = matrix_specs(
        "training", training_sizes, methodology["training_seed"]
    )
    held_specs = matrix_specs("held_out", held_out_sizes, methodology["held_out_seed"])
    if (
        type(report["training"]) is not list
        or type(report["held_out"]) is not list
        or len(report["training"]) != len(training_specs)
        or len(report["held_out"]) != len(held_specs)
    ):
        raise ValueError("matrix membership mismatch")
    for row, spec in zip(report["training"], training_specs, strict=True):
        phase, candidate, scenario, scenario_index, active_entries, seed_value = spec
        _verify_row(
            row,
            blocks,
            phase=phase,
            candidate=candidate,
            scenario=scenario,
            scenario_index=scenario_index,
            scenario_count=len(scenarios),
            active_entries=active_entries,
            seed_value=seed_value,
        )
    for row, spec in zip(report["held_out"], held_specs, strict=True):
        phase, candidate, scenario, scenario_index, active_entries, seed_value = spec
        _verify_row(
            row,
            blocks,
            phase=phase,
            candidate=candidate,
            scenario=scenario,
            scenario_index=scenario_index,
            scenario_count=len(scenarios),
            active_entries=active_entries,
            seed_value=seed_value,
        )
    _verify_row(
        report["default_linear_control"],
        blocks,
        phase="default_linear_control",
        candidate="default",
        scenario=SCENARIOS[0],
        scenario_index=0,
        scenario_count=len(SCENARIOS),
        active_entries=DEFAULT_LINEAR_CONTROL_SIZE,
        seed_value=DEFAULT_LINEAR_CONTROL_SEED,
    )

    construction_specs = [*training_specs, *held_specs]
    if type(report["construction"]) is not list or len(report["construction"]) != len(
        construction_specs
    ):
        raise ValueError("construction matrix membership mismatch")
    algorithms = {"projection": "axis_projection", "grid": "sparse_grid"}
    for row, spec in zip(report["construction"], construction_specs, strict=True):
        phase, candidate, scenario, _, active_entries, _ = spec
        if type(row) is not dict or set(row) != {
            "phase",
            "candidate",
            "scenario",
            "active_entries",
            "baseline",
            "candidate_metrics",
            "retained_memory_ratio",
        }:
            raise ValueError("construction schema mismatch")
        if (
            row["phase"] != phase
            or type(row["phase"]) is not str
            or row["candidate"] != candidate
            or type(row["candidate"]) is not str
            or not _exact(row["scenario"], dataclasses.asdict(scenario))
            or row["active_entries"] != active_entries
            or type(row["active_entries"]) is not int
        ):
            raise ValueError("construction matrix membership mismatch")
        for metrics, algorithm in (
            (row["baseline"], "linear"),
            (row["candidate_metrics"], algorithms[candidate]),
        ):
            if type(metrics) is not dict or set(metrics) != {
                "algorithm",
                "construction_ns",
                "retained_bytes",
                "rss_before_bytes",
                "rss_after_bytes",
                "rss_high_water_delta_bytes",
            }:
                raise ValueError("construction metrics schema mismatch")
            if (
                metrics["algorithm"] != algorithm
                or type(metrics["algorithm"]) is not str
                or any(
                    type(metrics[key]) is not int or metrics[key] < 0
                    for key in metrics
                    if key != "algorithm"
                )
                or metrics["construction_ns"] <= 0
                or metrics["retained_bytes"] <= 0
                or metrics["rss_after_bytes"] < metrics["rss_before_bytes"]
                or metrics["rss_high_water_delta_bytes"]
                != metrics["rss_after_bytes"] - metrics["rss_before_bytes"]
            ):
                raise ValueError("construction metric exact type/value mismatch")
        expected_ratio = (
            row["candidate_metrics"]["retained_bytes"]
            / row["baseline"]["retained_bytes"]
        )
        if (
            type(row["retained_memory_ratio"]) is not float
            or row["retained_memory_ratio"] != expected_ratio
        ):
            raise ValueError("retained memory ratio mismatch")
    expected_decision = _gate(
        report["training"],
        report["held_out"],
        report["construction"],
        methodology["training_sizes"],
        report["default_linear_control"],
    )
    if not _exact(report["decision"], expected_decision):
        raise ValueError("recomputed decision mismatch")
    expected_correctness = [
        _semantic_trace(kind) for kind in ("linear", *methodology["candidates"])
    ]
    if not _exact(report["correctness"], expected_correctness):
        raise ValueError("recomputed correctness mismatch")
    expected_adversaries = {
        "broad_query": {
            "scenario": dataclasses.asdict(
                next(item for item in SCENARIOS if item.name == "high-broad-both")
            ),
            "required_in_training": any(
                item.name == "high-broad-both" for item in scenarios
            ),
        },
        "grid_guard": _grid_guard_adversary(),
    }
    if not _exact(report["adversaries"], expected_adversaries):
        raise ValueError("recomputed adversary mismatch")
    if not _exact(report["provenance"], _provenance()):
        raise ValueError("source/runtime/backend provenance mismatch")
    if output.with_suffix(".md").read_text() != _markdown(report, digest):
        raise ValueError("Markdown mismatch")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=MINIMUM_BLOCKS)
    parser.add_argument("--profile", choices=("full", "focused"), default="full")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/experiments/radio-spectrum-index-matrix.json"),
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        report = verify_artifacts(args.output)
        print(f"verified {args.output}: {report['decision']['decision']}")
        return
    kwargs: dict[str, Any] = {}
    if args.profile == "focused":
        kwargs = {
            "training_sizes": TRAINING_SIZES[:2],
            "held_out_sizes": HELD_OUT_SIZES[:1],
            "scenarios": SCENARIOS[:1],
            "candidates": CANDIDATES[:1],
        }
    report = run_matrix(blocks=args.blocks, profile=args.profile, **kwargs)
    paths = write_artifacts(report, args.output)
    verify_artifacts(args.output)
    print(f"{report['decision']['decision']}: {', '.join(str(path) for path in paths)}")


if __name__ == "__main__":
    main()
