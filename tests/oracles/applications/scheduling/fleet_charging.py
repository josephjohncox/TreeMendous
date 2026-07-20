"""Independent integer charging feasibility reference."""


def duration(energy: int, power_per_slot: int) -> int:
    return (energy + power_per_slot - 1) // power_per_slot


def feasible(energy: int, power: int, dwell: int, bound: int) -> bool:
    slots = duration(energy, power)
    return slots <= dwell and slots <= bound
