"""Smoke benchmark for packet payload reassembly queries."""

from time import perf_counter

from treemendous.applications.catalogs.packet_sequence_reassembly import (
    PacketReassemblyCatalog,
)


def run_smoke(iterations: int = 500) -> float:
    catalog = PacketReassemblyCatalog()
    for index in range(200):
        catalog.add("flow", index * 8, b"abcdefgh")
    started = perf_counter()
    for index in range(iterations):
        start = (index % 190) * 8
        catalog.assemble("flow", start, start + 80)
    return perf_counter() - started


if __name__ == "__main__":
    print(run_smoke())
