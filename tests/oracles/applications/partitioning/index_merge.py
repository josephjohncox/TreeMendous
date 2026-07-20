"""Independent set/sort posting-list merge oracle."""

from collections.abc import Mapping, Sequence


def expected_index(
    segments: Sequence[Mapping[str, Sequence[int]]],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    terms = sorted({term for segment in segments for term in segment})
    return tuple(
        (
            term,
            tuple(
                sorted(
                    {
                        posting
                        for segment in segments
                        for posting in segment.get(term, ())
                    }
                )
            ),
        )
        for term in terms
    )
