"""Independent frame-chunk overlap reference."""


def chunks_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]
