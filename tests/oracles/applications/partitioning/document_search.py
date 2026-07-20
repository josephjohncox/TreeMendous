"""Independent exhaustive oracle for document search."""

import re
from collections.abc import Mapping


def expected_hits(documents: Mapping[int, str], query: str) -> tuple[int, ...]:
    tokens = tuple(dict.fromkeys(re.findall(r"[A-Za-z0-9]+", query.lower())))
    return tuple(
        key
        for key, text in sorted(documents.items())
        if set(tokens) <= set(re.findall(r"[A-Za-z0-9]+", text.lower()))
    )
