"""Deterministic generated-input fuzzing with crash deduplication and retries."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from dataclasses import dataclass

from treemendous.applications._shared.claiming import ClaimUnavailableError, WorkClaim
from treemendous.applications._shared.clock import Clock
from treemendous.applications.partitioning._runtime import PartitionRuntime, positive

FuzzTarget = Callable[[bytes], object]


@dataclass(frozen=True, order=True)
class Crash:
    """A deduplicated target failure with its first generated ordinal."""

    signature: str
    ordinal: int
    input: bytes
    exception_type: str
    message: str


@dataclass(frozen=True)
class FuzzingSnapshot:
    """Detached generated-input and crash state."""

    executed_ordinals: tuple[int, ...]
    crashes: tuple[Crash, ...]
    retries: int


class FuzzingEngine:
    """Generate ordinal-stable inputs and execute an injected local target.

    Target exceptions are findings and therefore complete their claims.
    Infrastructure failures explicitly abandon a claim for retry. Coordination
    is process-local; external workers need durable claims and fenced findings.
    """

    def __init__(
        self,
        target: FuzzTarget,
        *,
        cases: int,
        seed: int = 0,
        max_input_size: int = 32,
        clock: Clock | None = None,
    ) -> None:
        if not callable(target):
            raise TypeError("target must be callable")
        positive(cases, "cases")
        positive(max_input_size, "max_input_size")
        if type(seed) is not int:
            raise TypeError("seed must be an integer")
        self._target = target
        self._cases = cases
        self._seed = seed
        self._max_input_size = max_input_size
        self._executed: set[int] = set()
        self._crashes: dict[str, Crash] = {}
        self._retries = 0
        self._runtime = PartitionRuntime(cases, clock=clock)

    def input_for(self, ordinal: int) -> bytes:
        """Map an ordinal to bytes independently of worker or claim order."""
        if type(ordinal) is not int or not 0 <= ordinal < self._cases:
            raise ValueError("ordinal is outside the fuzzing domain")
        randomizer = random.Random((self._seed << 32) ^ ordinal)
        length = randomizer.randrange(self._max_input_size + 1)
        return bytes(randomizer.randrange(256) for _ in range(length))

    @staticmethod
    def _signature(exc: Exception) -> str:
        identity = f"{type(exc).__module__}.{type(exc).__qualname__}:{exc}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]

    def claim(self, owner: str, length: int) -> WorkClaim:
        """Claim generated-input ordinals."""
        return self._runtime.claim(owner, length)

    def execute_claim(
        self, claim: WorkClaim, *, infrastructure_failure: bool = False
    ) -> tuple[Crash, ...]:
        """Run target calls, or abandon the whole band on worker failure."""
        if infrastructure_failure:
            self._runtime.abandon(claim)
            self._retries += 1
            return ()
        found: list[Crash] = []
        for ordinal in range(claim.span.start, claim.span.end):
            data = self.input_for(ordinal)
            try:
                self._target(data)
            except (Exception,) as exc:
                signature = self._signature(exc)
                crash = Crash(signature, ordinal, data, type(exc).__name__, str(exc))
                prior = self._crashes.get(signature)
                if prior is None or ordinal < prior.ordinal:
                    self._crashes[signature] = crash
                found.append(crash)
            self._executed.add(ordinal)
        self._runtime.complete(claim, "executed", {"crashes": len(found)})
        return tuple(sorted(found))

    def run(
        self,
        *,
        shard_size: int = 64,
        fail_first_claim: bool = False,
    ) -> tuple[Crash, ...]:
        """Run every case; optionally exercise one abandoned-claim retry."""
        positive(shard_size, "shard_size")
        injected = fail_first_claim
        while True:
            try:
                claim = self.claim("local", shard_size)
            except ClaimUnavailableError:
                break
            self.execute_claim(claim, infrastructure_failure=injected)
            injected = False
        return tuple(sorted(self._crashes.values()))

    def snapshot(self) -> FuzzingSnapshot:
        """Return immutable execution, crash, and retry evidence."""
        return FuzzingSnapshot(
            tuple(sorted(self._executed)),
            tuple(sorted(self._crashes.values())),
            self._retries,
        )

    def checkpoint(self) -> tuple[FuzzingSnapshot, object]:
        """Capture findings and process-local coordination state."""
        return self.snapshot(), self._runtime.checkpoint()


def _default_target(data: bytes) -> None:
    if data.startswith(b"\x00"):
        raise ValueError("leading zero")


def create_fuzzing(
    target: FuzzTarget = _default_target,
    *,
    cases: int = 128,
    seed: int = 0,
    max_input_size: int = 32,
    clock: Clock | None = None,
) -> FuzzingEngine:
    """Create a deterministic fuzzing job."""
    return FuzzingEngine(
        target,
        cases=cases,
        seed=seed,
        max_input_size=max_input_size,
        clock=clock,
    )
