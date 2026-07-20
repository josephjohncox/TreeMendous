"""Record- and byte-split map/reduce with deterministic associative reduction."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from treemendous.applications._shared.claiming import ClaimUnavailableError, WorkClaim
from treemendous.applications._shared.clock import Clock
from treemendous.applications.partitioning._runtime import PartitionRuntime, positive

Mapper = Callable[[bytes], Iterable[tuple[str, Any]]]
Reducer = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class InputSplit:
    """One byte span containing either a raw chunk or complete records."""

    split_id: int
    start: int
    end: int
    units: tuple[bytes, ...]


@dataclass(frozen=True)
class MapReduceSnapshot:
    """Immutable split plan and reduced key/value output."""

    mode: str
    splits: tuple[InputSplit, ...]
    results: tuple[tuple[str, Any], ...]


class MapReduceEngine:
    """Map claimed input splits and reduce in stable split/emission order.

    ``reducer`` is required to be associative; deterministic ordering is
    guaranteed, but associativity cannot be inferred from arbitrary Python.
    All state is process-local. Distributed deployments must persist map output
    and fence commits before performing their deterministic final reduction.
    """

    def __init__(
        self,
        data: bytes,
        mapper: Mapper,
        reducer: Reducer,
        *,
        split_size: int,
        mode: str = "records",
        clock: Clock | None = None,
    ) -> None:
        if not isinstance(data, bytes) or not data:
            raise ValueError("data must be nonempty bytes")
        if not callable(mapper) or not callable(reducer):
            raise TypeError("mapper and reducer must be callable")
        positive(split_size, "split_size")
        if mode not in {"records", "bytes"}:
            raise ValueError("mode must be records or bytes")
        self._data = data
        self._mapper = mapper
        self._reducer = reducer
        self._mode = mode
        self._splits = self._make_splits(data, split_size, mode)
        self._mapped: dict[int, tuple[tuple[str, Any], ...]] = {}
        self._runtime = PartitionRuntime(len(self._splits), clock=clock)

    @staticmethod
    def _make_splits(data: bytes, size: int, mode: str) -> tuple[InputSplit, ...]:
        splits: list[InputSplit] = []
        if mode == "bytes":
            for split_id, start in enumerate(range(0, len(data), size)):
                end = min(len(data), start + size)
                splits.append(InputSplit(split_id, start, end, (data[start:end],)))
            return tuple(splits)
        records = data.splitlines(keepends=True)
        if not records:
            records = [data]
        cursor = 0
        for split_id, first in enumerate(range(0, len(records), size)):
            units = tuple(records[first : first + size])
            length = sum(len(unit) for unit in units)
            splits.append(InputSplit(split_id, cursor, cursor + length, units))
            cursor += length
        return tuple(splits)

    @property
    def splits(self) -> tuple[InputSplit, ...]:
        """Return the immutable record- or byte-split plan."""
        return self._splits

    def claim(self, owner: str, length: int) -> WorkClaim:
        """Claim contiguous split IDs."""
        return self._runtime.claim(owner, length)

    def execute_claim(self, claim: WorkClaim) -> tuple[tuple[str, Any], ...]:
        """Map every unit in a split band and commit only validated emissions."""

        def prepare() -> tuple[
            tuple[tuple[str, Any], ...], dict[int, tuple[tuple[str, Any], ...]]
        ]:
            staged: dict[int, tuple[tuple[str, Any], ...]] = {}
            try:
                for split in self._splits[claim.span.start : claim.span.end]:
                    emissions: list[tuple[str, Any]] = []
                    for unit in split.units:
                        for emission in self._mapper(unit):
                            if not isinstance(emission, tuple) or len(emission) != 2:
                                raise TypeError("mapper must emit key/value pairs")
                            key, value = emission
                            if not isinstance(key, str) or not key:
                                raise ValueError("mapper keys must be nonempty strings")
                            emissions.append((key, value))
                    staged[split.split_id] = tuple(emissions)
            except (Exception,) as exc:
                raise RuntimeError("mapper execution failed") from exc
            mapped = self._mapped.copy()
            mapped.update(staged)
            output = tuple(value for key in sorted(staged) for value in staged[key])
            return output, mapped

        prepared = self._runtime.execute_claim(
            claim,
            kind="mapped",
            prepare=prepare,
            commit=lambda value: setattr(self, "_mapped", value[1]),
            result=lambda value: {"emissions": len(value[0])},
        )
        return prepared[0]

    def _reduce(self) -> tuple[tuple[str, Any], ...]:
        grouped: dict[str, list[Any]] = {}
        for split_id in sorted(self._mapped):
            for key, value in self._mapped[split_id]:
                grouped.setdefault(key, []).append(value)
        reduced: list[tuple[str, Any]] = []
        for key in sorted(grouped):
            values = grouped[key]
            accumulator = values[0]
            for value in values[1:]:
                accumulator = self._reducer(accumulator, value)
            reduced.append((key, accumulator))
        return tuple(reduced)

    def reduce(self) -> tuple[tuple[str, Any], ...]:
        """Reduce by key in stable split and mapper-emission order."""
        return self._runtime.observe(self._reduce)

    def run(self, *, shard_size: int = 8) -> tuple[tuple[str, Any], ...]:
        """Map all remaining splits and return deterministic reduction."""
        positive(shard_size, "shard_size")
        while True:
            try:
                claim = self.claim("local", shard_size)
            except ClaimUnavailableError:
                break
            self.execute_claim(claim)
        return self.reduce()

    def _snapshot(self) -> MapReduceSnapshot:
        return MapReduceSnapshot(self._mode, self._splits, self._reduce())

    def snapshot(self) -> MapReduceSnapshot:
        """Return immutable plan and current reduction."""
        return self._runtime.observe(self._snapshot)

    def audit_snapshot(self) -> tuple[MapReduceSnapshot, object]:
        """Capture non-restorable application and runtime audit evidence."""
        return self._runtime.audit_snapshot(self._snapshot)


def _word_mapper(unit: bytes) -> Iterable[tuple[str, int]]:
    for word in unit.decode("utf-8").split():
        yield word.lower(), 1


def _sum(left: Any, right: Any) -> Any:
    return left + right


def create_map_reduce(
    data: bytes = b"range trees\nrange sets\n",
    mapper: Mapper = _word_mapper,
    reducer: Reducer = _sum,
    *,
    split_size: int = 1,
    mode: str = "records",
    clock: Clock | None = None,
) -> MapReduceEngine:
    """Create a deterministic input-split map/reduce job."""
    return MapReduceEngine(
        data,
        mapper,
        reducer,
        split_size=split_size,
        mode=mode,
        clock=clock,
    )
