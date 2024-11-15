from typing import Optional, List, overload

class IntervalNode:
    def __init__(self, start: int, end: int) -> None:
        self.start: int = start
        self.end: int = end
        self.length: int = end - start
        self.total_length: int = self.length
        
        self.left: Optional[IntervalNode] = None
        self.right: Optional[IntervalNode] = None
        self.height: int = 1

    def update_stats(self) -> None:
        self.length = self.end - self.start
        self.total_length = self.length
        if self.left:
            self.total_length += self.left.total_length
        if self.right:
            self.total_length += self.right.total_length

        self.height = 1 + max(self.get_height(self.left), self.get_height(self.right))

    @staticmethod
    def get_height(node: Optional['IntervalNode']) -> int:
        if not node:
            return 0
        return node.height

class IntervalTree:
    def __init__(self) -> None:
        self.root: Optional[IntervalNode] = None

    def insert_interval(self, start: int, end: int) -> None:
        overlapping_nodes: List[IntervalNode] = []
        self.root = self._delete_overlaps(self.root, start, end, overlapping_nodes)
        # Merge overlapping intervals with the new interval
        for node in overlapping_nodes:
            start = min(start, node.start)
            end = max(end, node.end)
        # Insert the merged interval
        self.root = self._insert(self.root, IntervalNode(start, end))

    def delete_interval(self, start: int, end: int) -> None:
        # Find overlapping intervals
        overlapping_nodes: List[IntervalNode] = []
        self.root = self._delete_overlaps(self.root, start, end, overlapping_nodes)
        # For each overlapping interval, we may need to split it
        for node in overlapping_nodes:
            if node.start < start:
                # Left part remains available
                self.root = self._insert(self.root, IntervalNode(node.start, start))
            if node.end > end:
                # Right part remains available
                self.root = self._insert(self.root, IntervalNode(end, node.end))

    def _delete_overlaps(self, node: Optional[IntervalNode], start: int, end: int, 
                        overlapping_nodes: List[IntervalNode]) -> Optional[IntervalNode]:
        if not node:
            return None

        # If the current node interval is completely after the interval to delete, go left
        if end <= node.start:
            node.left = self._delete_overlaps(node.left, start, end, overlapping_nodes)
        # If the current node interval is completely before the interval to delete, go right
        elif start >= node.end:
            node.right = self._delete_overlaps(node.right, start, end, overlapping_nodes)
        else:
            # Current node overlaps with [start, end], remove it and collect it
            overlapping_nodes.append(node)
            # Delete this node and replace it with its children
            if node.left and node.right:
                # Node with two children: Get the inorder successor (smallest in the right subtree)
                successor = self._get_min(node.right)
                # Copy the successor's content to this node
                node.start = successor.start
                node.end = successor.end
                # Delete the successor
                node.right = self._delete_overlaps(node.right, successor.start, successor.end, overlapping_nodes)
            elif node.left:
                node = node.left
            else:
                node = node.right

        if node:
            node.update_stats()
            node = self._rebalance(node)
        return node

    def _insert(self, node: Optional[IntervalNode], new_node: IntervalNode) -> IntervalNode:
        if not node:
            return new_node

        if new_node.start < node.start:
            node.left = self._insert(node.left, new_node)
        else:
            node.right = self._insert(node.right, new_node)

        node.update_stats()
        node = self._rebalance(node)
        return node

    def _get_min(self, node: IntervalNode) -> IntervalNode:
        current = node
        while current.left:
            current = current.left
        return current

    def _rebalance(self, node: IntervalNode) -> IntervalNode:
        balance = self._get_balance(node)
        if balance > 1:
            # Left heavy
            if self._get_balance(node.left) < 0:
                # Left-Right case
                node.left = self._rotate_left(node.left)
            # Left-Left case
            node = self._rotate_right(node)
        elif balance < -1:
            # Right heavy
            if self._get_balance(node.right) > 0:
                # Right-Left case
                node.right = self._rotate_right(node.right)
            # Right-Right case
            node = self._rotate_left(node)
        return node

    def _get_balance(self, node: Optional[IntervalNode]) -> int:
        if not node:
            return 0
        return IntervalNode.get_height(node.left) - IntervalNode.get_height(node.right)

    @overload
    def _rotate_left(self, z: None) -> None: ...

    @overload
    def _rotate_left(self, z: IntervalNode) -> IntervalNode: ...

    def _rotate_left(self, z: Optional[IntervalNode]) -> Optional[IntervalNode]:
        if not z or not z.right:
            return z
        y: IntervalNode = z.right
        subtree: Optional[IntervalNode] = y.left

        # Perform rotation
        y.left = z
        z.right = subtree

        # Update heights and stats
        z.update_stats()
        y.update_stats()
        return y

    @overload
    def _rotate_right(self, z: None) -> None: ...

    @overload
    def _rotate_right(self, z: IntervalNode) -> IntervalNode: ...

    def _rotate_right(self, z: Optional[IntervalNode]) -> Optional[IntervalNode]:
        if not z or not z.left:
            return z
        y: IntervalNode = z.left
        subtree: Optional[IntervalNode] = y.right

        # Perform rotation
        y.right = z
        z.left = subtree

        # Update heights and stats
        z.update_stats()
        y.update_stats()
        return y

    def get_total_available_length(self) -> int:
        if not self.root:
            return 0
        return self.root.total_length

    def print_tree(self) -> None:
        self._print_tree(self.root)

    def _print_tree(self, node: Optional[IntervalNode], indent: str = "", prefix: str = "") -> None:
        if node is None:
            return
            
        # Print right subtree
        self._print_tree(node.right, indent + "    ", "┌── ")
        
        # Print current node
        print(f"{indent}{prefix}{node.start}-{node.end} (len={node.length}, total_len={node.total_length})")
        
        # Print left subtree  
        self._print_tree(node.left, indent + "    ", "└── ")

# Example usage:
if __name__ == "__main__":
    tree = IntervalTree()
    # Initially, the whole interval [0, 100] is available
    tree.insert_interval(0, 100)
    print("Initial tree:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Schedule interval [10, 20]
    tree.delete_interval(10, 20)
    print("\nAfter scheduling [10, 20]:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Schedule interval [30, 40]
    tree.delete_interval(30, 40)
    print("\nAfter scheduling [30, 40]:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Unschedule interval [10, 20]
    tree.insert_interval(10, 20)
    print("\nAfter unscheduling [10, 20]:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")

    # Split at pivot 50 (delete [50, 50])
    tree.delete_interval(50, 50)
    print("\nAfter splitting at pivot 50:")
    tree.print_tree()
    print(f"Total available length: {tree.get_total_available_length()}")