"""CNF SAT search partitioned by assignment-prefix ordinal."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from treemendous.applications._shared.claiming import ClaimUnavailableError, WorkClaim
from treemendous.applications._shared.clock import Clock
from treemendous.applications.partitioning._runtime import PartitionRuntime, positive


@dataclass(frozen=True, order=True)
class SatisfyingAssignment:
    """One satisfying full assignment in integer ordinal order."""

    ordinal: int
    values: tuple[bool, ...]


@dataclass(frozen=True)
class SatSearchSnapshot:
    """Detached satisfying assignments accumulated so far."""

    variables: int
    prefix_bits: int
    solutions: tuple[SatisfyingAssignment, ...]


class SatSearchEngine:
    """Enumerate assignment-prefix ordinals and evaluate a validated CNF.

    This is an exact finite in-memory solver, not a cluster scheduler. Claims
    are process-local; a distributed solution sink must durably fence writes.
    """

    def __init__(
        self,
        variables: int,
        clauses: object,
        *,
        prefix_bits: int | None = None,
        clock: Clock | None = None,
    ) -> None:
        positive(variables, "variables")
        if isinstance(clauses, (str, bytes)) or not isinstance(clauses, Sequence):
            raise TypeError("clauses must be a sequence")
        checked: list[tuple[int, ...]] = []
        for raw_clause in cast(Sequence[object], clauses):
            if isinstance(raw_clause, (str, bytes)) or not isinstance(
                raw_clause, Sequence
            ):
                raise TypeError("each clause must be a sequence")
            clause = cast(Sequence[object], raw_clause)
            if not clause:
                raise ValueError("clauses must not be empty")
            literals: list[int] = []
            for literal in clause:
                if type(literal) is not int or literal == 0 or abs(literal) > variables:
                    raise ValueError("literals must identify a signed variable")
                literals.append(literal)
            checked.append(tuple(literals))
        if not checked:
            raise ValueError("CNF must contain at least one clause")
        selected_prefix = min(variables, 8) if prefix_bits is None else prefix_bits
        if type(selected_prefix) is not int or not 0 <= selected_prefix <= variables:
            raise ValueError("prefix_bits must be between zero and variables")
        self._variables = variables
        self._clauses = tuple(checked)
        self._prefix_bits = selected_prefix
        self._solutions: dict[int, SatisfyingAssignment] = {}
        self._runtime = PartitionRuntime(1 << selected_prefix, clock=clock)

    def claim(self, owner: str, length: int) -> WorkClaim:
        """Claim assignment-prefix ordinals."""
        return self._runtime.claim(owner, length)

    def _satisfies(self, ordinal: int) -> bool:
        return all(
            any(
                bool(ordinal & (1 << (abs(literal) - 1))) == (literal > 0)
                for literal in clause
            )
            for clause in self._clauses
        )

    def evaluate_claim(self, claim: WorkClaim) -> tuple[SatisfyingAssignment, ...]:
        """Enumerate suffixes beneath every claimed prefix and evaluate CNF."""
        def prepare() -> tuple[
            tuple[SatisfyingAssignment, ...], dict[int, SatisfyingAssignment]
        ]:
            suffix_bits = self._variables - self._prefix_bits
            found: list[SatisfyingAssignment] = []
            solutions = self._solutions.copy()
            for prefix in range(claim.span.start, claim.span.end):
                for suffix in range(1 << suffix_bits):
                    ordinal = prefix | (suffix << self._prefix_bits)
                    if self._satisfies(ordinal):
                        values = tuple(
                            bool(ordinal & (1 << variable))
                            for variable in range(self._variables)
                        )
                        solution = SatisfyingAssignment(ordinal, values)
                        solutions[ordinal] = solution
                        found.append(solution)
            return tuple(sorted(found)), solutions

        prepared = self._runtime.execute_claim(
            claim,
            kind="evaluated",
            prepare=prepare,
            commit=lambda value: setattr(self, "_solutions", value[1]),
            result=lambda value: {"solutions": len(value[0])},
        )
        return prepared[0]

    def run(self, *, shard_size: int = 32) -> tuple[SatisfyingAssignment, ...]:
        """Evaluate all prefixes and return solutions in ordinal order."""
        positive(shard_size, "shard_size")
        while True:
            try:
                claim = self.claim("local", shard_size)
            except ClaimUnavailableError:
                break
            self.evaluate_claim(claim)
        return tuple(self._solutions[key] for key in sorted(self._solutions))

    def _snapshot(self) -> SatSearchSnapshot:
        return SatSearchSnapshot(
            self._variables,
            self._prefix_bits,
            tuple(self._solutions[key] for key in sorted(self._solutions)),
        )

    def snapshot(self) -> SatSearchSnapshot:
        """Return immutable solver state."""
        return self._runtime.observe(self._snapshot)

    def audit_snapshot(self) -> tuple[SatSearchSnapshot, object]:
        """Capture non-restorable application and runtime audit evidence."""
        return self._runtime.audit_snapshot(self._snapshot)


def create_sat_search(
    variables: int = 3,
    clauses: Sequence[Sequence[int]] = ((1, 2), (-1, 3)),
    *,
    prefix_bits: int | None = None,
    clock: Clock | None = None,
) -> SatSearchEngine:
    """Create an exact partitioned CNF search job."""
    return SatSearchEngine(
        variables, clauses, prefix_bits=prefix_bits, clock=clock
    )
