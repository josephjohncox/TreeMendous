"""Independent sequential replay oracle."""

from collections.abc import Iterable


def expected_state(events: Iterable[tuple[int, str, str, int | str | None]]) -> tuple[tuple[str, int | str], ...]:
    state: dict[str, int | str] = {}
    for _, key, operation, value in sorted(events):
        if operation == "delete":
            state.pop(key, None)
        elif operation == "set":
            if not isinstance(value, (int, str)):
                raise TypeError("invalid oracle set value")
            state[key] = value
        else:
            current = state.get(key, 0)
            if not isinstance(value, int) or not isinstance(current, int):
                raise TypeError("invalid oracle increment value")
            state[key] = current + value
    return tuple(sorted(state.items()))
