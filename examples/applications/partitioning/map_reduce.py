#!/usr/bin/env python3
"""Run record-split map/reduce from any working directory."""

from treemendous.applications.partitioning.map_reduce import MapReduceEngine


def mapper(unit: bytes):
    return ((word, 1) for word in unit.decode().split())


def reduce_sum(left: int, right: int) -> int:
    return left + right


def main() -> None:
    result = MapReduceEngine(b"a b\na\n", mapper, reduce_sum, split_size=1).run()
    if result != (("a", 2), ("b", 1)):
        raise RuntimeError("unexpected map/reduce result")
    print("map-reduce: a=2,b=1")


if __name__ == "__main__":
    main()
