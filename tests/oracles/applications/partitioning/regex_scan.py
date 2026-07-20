"""Independent whole-buffer oracle for byte regex scanning."""

import re


def expected_matches(data: bytes, pattern: bytes) -> tuple[tuple[int, int, bytes], ...]:
    return tuple((match.start(), match.end(), match.group()) for match in re.finditer(pattern, data))
