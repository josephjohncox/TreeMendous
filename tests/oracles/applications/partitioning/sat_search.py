"""Independent exhaustive truth-table oracle for CNF SAT."""

from collections.abc import Sequence


def expected_ordinals(variables: int, clauses: Sequence[Sequence[int]]) -> tuple[int, ...]:
    result = []
    for ordinal in range(2**variables):
        if all(any(bool(ordinal & (2 ** (abs(lit) - 1))) == (lit > 0) for lit in clause) for clause in clauses):
            result.append(ordinal)
    return tuple(result)
