from abc import ABC, abstractmethod
from typing import Callable, Generic, List, Optional, Tuple, TypeVar, Protocol

D = TypeVar('D')  # Type variable for interval data

class IntervalNodeProtocol(Protocol[D]):
    start: int
    end: int
    length: int
    height: int
    total_length: int
    data: Optional[D]
    left: Optional['IntervalNodeProtocol[D]']
    right: Optional['IntervalNodeProtocol[D]']
    
    def update_stats(self) -> None: ...
    def update_length(self) -> None: ...

T = TypeVar('T', bound='IntervalNodeProtocol[D]')

class IntervalManagerProtocol(Protocol[D]):
    def reserve_interval(self, start: int, end: int, data: Optional[D] = None) -> None: ...
    def release_interval(self, start: int, end: int, data: Optional[D] = None) -> None: ...
    def find_interval(self, start: int, length: int) -> Tuple[int, int]: ...
    def get_intervals(self) -> List[Tuple[int, int, Optional[D]]]: ...

class IntervalNodeBase(Generic[T, D]):
    def __init__(self, start: int, end: int, data: Optional[D] = None) -> None:
        self.start: int = start
        self.end: int = end
        self.length: int = end - start
        self._height: int = 1
        self._total_length: int = self.length
        self.data: Optional[D] = data

        self.left: Optional[T] = None
        self.right: Optional[T] = None

    @property
    @abstractmethod
    def height(self) -> int:
        return self._height

    @height.setter 
    def height(self, value: int) -> None:
        self._height = value

    @property
    @abstractmethod
    def total_length(self) -> int:
        return self._total_length

    @total_length.setter 
    def total_length(self, value: int) -> None:
        self._total_length = value

    def update_length(self) -> None:
        self.length = self.end - self.start


class IntervalTreeBase(Generic[T, D], ABC, IntervalManagerProtocol[D]):
    def __init__(self, root: Optional[T] = None, merge_fn: Optional[Callable[[D, D], D]] = None) -> None:
        self.root: Optional[T] = root
        self.merge_fn: Optional[Callable[[D, D], D]] = merge_fn

    def print_tree(self) -> None:
        self._print_tree(self.root)

    def _print_tree(self, node: Optional[T], indent: str = "", prefix: str = "") -> None:
        if node is None:
            return
            
        self._print_tree(node.right, indent + "    ", "┌── ")  # type: ignore
        self._print_node(node, indent, prefix)
        if node.data is not None:
            print(f"{indent}{prefix}data: {node.data}")
        self._print_tree(node.left, indent + "    ", "└── ")   # type: ignore

    def get_total_available_length(self) -> int:
        if not self.root:
            return 0
        return self.root.total_length

    def merge_data(self, data1: Optional[D], data2: Optional[D]) -> Optional[D]:
        if data1 is None:
            return data2
        if data2 is None:
            return data1
        if self.merge_fn is None:
            # Default behavior: treat data as sets and merge via union
            if isinstance(data1, set) and isinstance(data2, set):
                return data1 | data2
            return data1  # If not sets, keep first value
        return self.merge_fn(data1, data2)

    @abstractmethod
    def _print_node(self, node: T, indent: str, prefix: str) -> None: ...

    @abstractmethod
    def get_intervals(self) -> List[Tuple[int, int, Optional[D]]]: ...
