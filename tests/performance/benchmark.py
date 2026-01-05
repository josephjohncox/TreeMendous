from typing import List, Tuple

from tests.performance.workload import generate_realistic_workload

def generate_operations(num_operations: int) -> List[Tuple[str, int, int]]:
    return generate_realistic_workload(
        num_operations=num_operations,
        profile="scheduler",
        space_range=(0, 999_900),
        operation_mix={'reserve': 0.4, 'release': 0.4, 'find': 0.2},
        seed=42,
        include_data=False
    )
