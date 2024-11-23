from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

T = TypeVar('T', bound='IntervalNodeBase')

class IntervalNodeBase(Generic[T]):
    def __init__(self, start: int, end: int) -> None:
        self.start: int = start
        self.end: int = end
        self.length: int = end - start
        self._height: int = 1
        self._total_length: int = self.length

        self.left: Optional[T] = None
        self.right: Optional[T] = None

    def update_length(self) -> None:
        self.length = self.end - self.start

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

    @staticmethod
    def get_height(node: Optional[T]) -> int:
        if not node:
            return 0
        return node.height


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

