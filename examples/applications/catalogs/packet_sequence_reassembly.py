"""Assemble first-arrival packet payload across fragments."""

from treemendous.applications.catalogs.packet_sequence_reassembly import (
    PacketReassemblyCatalog,
)


def main() -> None:
    catalog = PacketReassemblyCatalog()
    catalog.add("connection", 0, b"Tree")
    catalog.add("connection", 4, b"-Mendous")
    print(catalog.assemble("connection", 0, 12).payload)


if __name__ == "__main__":
    main()
