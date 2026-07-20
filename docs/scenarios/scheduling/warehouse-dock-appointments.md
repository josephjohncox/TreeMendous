# Warehouse dock appointments

`WarehouseDockScheduler` models exclusive docks with cargo compatibility labels.
Appointments have a bounded integer search window and pre/post handling buffers.
Buffers extend occupied time while preserving the visible carrier appointment,
so touching is permitted only at occupied boundaries. The engine chooses
integer start then dock name deterministically.

Compatibility or capacity failure changes no state. Temporal failure exposes
conflict diagnostics with the dock, exact blocked segment, and reservation IDs.
Owner-scoped request IDs replay an identical appointment and reject changed
input. Cancellation checks the carrier owner, is idempotent, and restores dock
capacity. Snapshots are immutable and deterministically ordered.

The module does not plan labor, yard queues, door dimensions, inventory, travel
time, demurrage, or optimal dock utilization. Cargo types are literal caller
labels. It is an in-memory feasibility component rather than a warehouse
management system: process loss discards state and no distributed fencing or
durable transaction is implied.

Run `python examples/applications/scheduling/warehouse_docks.py`.
