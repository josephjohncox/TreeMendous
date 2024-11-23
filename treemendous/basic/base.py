from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Tuple, TypeVar, Protocol

class IntervalNodeProtocol(Protocol):
    start: int
    end: int
    length: int
    height: int
    total_length: int
    left: Optional['IntervalNodeProtocol']
    right: Optional['IntervalNodeProtocol']
    
    def update_stats(self) -> None: ...
    def update_length(self) -> None: ...

T = TypeVar('T', bound=IntervalNodeProtocol)

class IntervalNodeBase(Generic[T]):
    def __init__(self, start: int, end: int) -> None:
        self.start: int = start
        self.end: int = end
        self.length: int = end - start
        self._height: int = 1
        self._total_length: int = self.length

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


class IntervalTreeBase(Generic[T], ABC):
    def __init__(self, root: Optional[T] = None) -> None:
        self.root: Optional[T] = root

    def print_tree(self) -> None:
        self._print_tree(self.root)

    def _print_tree(self, node: Optional[T], indent: str = "", prefix: str = "") -> None:
        if node is None:
            return
            
        self._print_tree(node.right, indent + "    ", "┌── ")  # type: ignore
        self._print_node(node, indent, prefix)
        self._print_tree(node.left, indent + "    ", "└── ")   # type: ignore

    def get_total_available_length(self) -> int:
        if not self.root:
            return 0
        return self.root.total_length

    @abstractmethod
    def _print_node(self, node: T, indent: str, prefix: str) -> None: ...

    @abstractmethod
    def get_all_intervals(self) -> List[Tuple[int, int]]: ...

