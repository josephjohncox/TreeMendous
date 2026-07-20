"""Result type for deterministic actual-work benchmark smokes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SmokeResult:
    operations: int
    oracle_checks: int
    elapsed_seconds: float

    @property
    def operations_per_second(self) -> float:
        return self.operations / self.elapsed_seconds if self.elapsed_seconds else 0.0
