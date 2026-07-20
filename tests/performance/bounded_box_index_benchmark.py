"""Correctness-checked bounded-index timing without a universal ranking claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from tests.oracles.multidimensional.brute_box_index import BruteBoxIndex
from treemendous.multidimensional import BoundedBoxIndex, Box

_BOUNDS = Box((0, 0, 0), (12, 12, 12))


def _exercise_oracle() -> tuple[
    tuple[tuple[tuple[int, str], ...], ...],
    tuple[tuple[int, tuple[int, ...], tuple[int, ...], str], ...],
]:
    oracle = BruteBoxIndex(3)
    first = oracle.insert((0, 0, 0), (3, 3, 3), "first")
    second = oracle.insert((0, 0, 0), (3, 3, 3), "duplicate")
    oracle.insert((5, 5, 5), (9, 9, 9), "far")
    oracle.insert((1, 8, 1), (11, 9, 2), "skew")
    traces = [
        tuple(
            (entry.handle, entry.data)
            for entry in oracle.overlaps((1, 1, 1), (2, 2, 2))
        )
    ]
    oracle.update(
        second,
        lower=(2, 2, 2),
        upper=(6, 6, 6),
        data="duplicate-updated",
    )
    traces.append(
        tuple(
            (entry.handle, entry.data)
            for entry in oracle.overlaps((4, 4, 4), (7, 7, 7))
        )
    )
    oracle.remove(first)
    oracle.insert((9, 0, 0), (12, 2, 2), "edge")
    traces.append(
        tuple(
            (entry.handle, entry.data)
            for entry in oracle.overlaps((10, 0, 0), (11, 1, 1))
        )
    )
    state = tuple(
        (entry.handle, entry.lower, entry.upper, entry.data)
        for entry in oracle.entries.values()
    )
    return tuple(traces), state


def _exercise_index(
    index: BoundedBoxIndex,
) -> tuple[tuple[tuple[int, str], ...], ...]:
    first = index.insert(Box((0, 0, 0), (3, 3, 3)), "first")
    second = index.insert(Box((0, 0, 0), (3, 3, 3)), "duplicate")
    index.insert(Box((5, 5, 5), (9, 9, 9)), "far")
    index.insert(Box((1, 8, 1), (11, 9, 2)), "skew")
    traces = [
        tuple(
            (entry.handle.sequence, entry.data)
            for entry in index.overlaps(Box((1, 1, 1), (2, 2, 2)))
        )
    ]
    index.update(
        second,
        box=Box((2, 2, 2), (6, 6, 6)),
        data="duplicate-updated",
    )
    traces.append(
        tuple(
            (entry.handle.sequence, entry.data)
            for entry in index.overlaps(Box((4, 4, 4), (7, 7, 7)))
        )
    )
    index.remove(first)
    index.insert(Box((9, 0, 0), (12, 2, 2)), "edge")
    traces.append(
        tuple(
            (entry.handle.sequence, entry.data)
            for entry in index.overlaps(Box((10, 0, 0), (11, 1, 1)))
        )
    )
    return tuple(traces)


def _final_state(
    index: BoundedBoxIndex,
) -> tuple[tuple[int, tuple[int, ...], tuple[int, ...], str], ...]:
    return tuple(
        (entry.handle.sequence, entry.box.lower, entry.box.upper, entry.data)
        for entry in index.entries()
    )


def run_benchmark(trials: int) -> dict[str, Any]:
    if trials <= 0:
        raise ValueError("trials must be greater than zero")
    expected_trace, expected_state = _exercise_oracle()
    durations: list[int] = []
    for _ in range(trials):
        index = BoundedBoxIndex(_BOUNDS, (2, 2, 2))
        started = perf_counter_ns()
        observed_trace = _exercise_index(index)
        elapsed = perf_counter_ns() - started

        # Correctness checks and final-state materialization are intentionally
        # outside the measured section for every timed index instance.
        observed_state = _final_state(index)
        if observed_trace != expected_trace:
            raise AssertionError("timed query trace differs from point-set oracle")
        if observed_state != expected_state:
            raise AssertionError("timed final state differs from point-set oracle")
        durations.append(elapsed)

    ordered_durations = sorted(durations)
    middle = len(ordered_durations) // 2
    if len(ordered_durations) % 2:
        median_duration_ns = ordered_durations[middle]
    else:
        median_duration_ns = (
            ordered_durations[middle - 1] + ordered_durations[middle]
        ) // 2

    return {
        "benchmark": "bounded_box_index_correctness_checked",
        "algorithm": "sparse_grid",
        "dimensions": 3,
        "trials": trials,
        "duration_ns": durations,
        "median_duration_ns": median_duration_ns,
        "correctness": "every timed instance matched an independent finite point-set oracle outside timing",
        "universal_ranking_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_benchmark(arguments.trials)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered)


if __name__ == "__main__":
    main()
