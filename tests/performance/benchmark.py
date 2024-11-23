from typing import List, Tuple
import random

def generate_operations(num_operations: int) -> List[Tuple[str, int, int]]:
    operations: List[Tuple[str, int, int]] = []
    for _ in range(num_operations):
        op_type: str = random.choice(['reserve', 'release', 'find'])
        start: int = random.randint(0, 999_900)
        end: int = start + random.randint(1, 100)
        operations.append((op_type, start, end))
    return operations