"""Independent deterministic-input and crash-signature oracle for fuzzing."""

import hashlib
import random


def generated(seed: int, ordinal: int, maximum: int) -> bytes:
    source = random.Random((seed << 32) ^ ordinal)
    return bytes(source.randrange(256) for _ in range(source.randrange(maximum + 1)))


def signature(exc: Exception) -> str:
    identity = f"{type(exc).__module__}.{type(exc).__qualname__}:{exc}"
    return hashlib.sha256(identity.encode()).hexdigest()[:20]
