"""Packet sequence reassembly contract."""

from tests.oracles.applications.catalogs.packet_sequence_reassembly import assemble
from treemendous.applications.catalogs.packet_sequence_reassembly import (
    PacketReassemblyCatalog,
)


def test_reassembly_retains_duplicates_gaps_and_first_arrival_payload() -> None:
    catalog = PacketReassemblyCatalog()
    first = catalog.add("flow", 0, b"abc")
    duplicate = catalog.add("flow", 1, b"ZZ")
    tail = catalog.add("flow", 4, b"e")
    result = catalog.assemble("flow", 0, 5)
    expected_payload, missing = assemble([(0, b"abc"), (1, b"ZZ"), (4, b"e")], 0, 5)
    assert result.payload == expected_payload is None
    assert (
        tuple(point for gap in result.gaps for point in range(gap.start, gap.end))
        == missing
    )
    assert result.duplicate_coverage
    catalog.update(tail, sequence=3, payload=b"de")
    assert catalog.assemble("flow", 0, 5).payload == b"abcde"
    assert catalog.remove(duplicate).handle == duplicate
    assert catalog.snapshot().records[0].handle == first
