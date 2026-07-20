#!/usr/bin/env python3
"""Run partitioned exact CNF search from any working directory."""

from treemendous.applications.partitioning.sat_search import SatSearchEngine


def main() -> None:
    solutions = SatSearchEngine(2, ((1,), (2,)), prefix_bits=1).run(shard_size=1)
    if tuple(item.ordinal for item in solutions) != (3,):
        raise RuntimeError("unexpected SAT result")
    print("sat-search: assignment 3 satisfies CNF")


if __name__ == "__main__":
    main()
