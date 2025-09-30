#!/usr/bin/env python3
"""
Treap Implementation for Interval Trees

Demonstrates randomized tree-heap (treap) structure for interval management
with probabilistic balancing through random priorities.
"""

import random
import sys
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass

# Add Tree-Mendous to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'treemendous' / 'basic'))

from base import IntervalNodeBase


@dataclass
class TreapInterval:
    """Interval with random priority for treap structure"""
    start: int
    end: int
    priority: float
    data: Optional[str] = None
    
    def __post_init__(self):
        if self.priority is None:
            self.priority = random.random()
    
    @property
    def length(self) -> int:
        return self.end - self.start


class TreapNode(IntervalNodeBase['TreapNode', str]):
    """Treap node combining BST and heap properties"""
    
    def __init__(self, start: int, end: int, data: Optional[str] = None, priority: Optional[float] = None):
        super().__init__(start, end, data)
        self.priority = priority if priority is not None else random.random()
        self.height = 1
        self.total_length = self.length
        
    def update_stats(self) -> None:
        """Update height and total length statistics"""
        self.update_length()
        self.total_length = self.length
        
        if self.left:
            self.total_length += self.left.total_length
        if self.right:
            self.total_length += self.right.total_length
            
        self.height = 1 + max(
            self.get_height(self.left),
            self.get_height(self.right)
        )
    
    @staticmethod
    def get_height(node: Optional['TreapNode']) -> int:
        return node.height if node else 0


class RandomizedIntervalTreap:
    """Treap-based interval tree with probabilistic balancing"""
    
    def __init__(self):
        self.root: Optional[TreapNode] = None
        self.size = 0
    
    def insert(self, start: int, end: int, data: Optional[str] = None) -> None:
        """Insert interval with random priority"""
        new_node = TreapNode(start, end, data)
        self.root = self._insert(self.root, new_node)
        self.size += 1
    
    def _insert(self, node: Optional[TreapNode], new_node: TreapNode) -> TreapNode:
        """Insert maintaining BST and heap properties"""
        if not node:
            return new_node
            
        # BST insertion by start time
        if new_node.start < node.start:
            node.left = self._insert(node.left, new_node)
            # Rotate right if heap property violated
            if node.left and node.left.priority > node.priority:
                node = self._rotate_right(node)
        else:
            node.right = self._insert(node.right, new_node)
            # Rotate left if heap property violated  
            if node.right and node.right.priority > node.priority:
                node = self._rotate_left(node)
        
        if node:
            node.update_stats()
        return node
    
    def delete(self, start: int, end: int) -> bool:
        """Delete interval from treap"""
        self.root, deleted = self._delete(self.root, start, end)
        if deleted:
            self.size -= 1
        return deleted
    
    def _delete(self, node: Optional[TreapNode], start: int, end: int) -> Tuple[Optional[TreapNode], bool]:
        """Delete maintaining treap properties"""
        if not node:
            return None, False
            
        if start < node.start:
            node.left, deleted = self._delete(node.left, start, end)
        elif start > node.start or end != node.end:
            node.right, deleted = self._delete(node.right, start, end)
        else:
            # Found node to delete - push it down using rotations
            if not node.left:
                return node.right, True
            elif not node.right:
                return node.left, True
            else:
                # Rotate with child having higher priority
                if node.left.priority > node.right.priority:
                    node = self._rotate_right(node)
                    node.right, deleted = self._delete(node.right, start, end)
                else:
                    node = self._rotate_left(node)
                    node.left, deleted = self._delete(node.left, start, end)
                    
        if node:
            node.update_stats()
        return node, deleted
    
    def _rotate_left(self, node: TreapNode) -> TreapNode:
        """Left rotation maintaining priorities"""
        if not node.right:
            return node
            
        new_root = node.right
        node.right = new_root.left
        new_root.left = node
        
        node.update_stats()
        new_root.update_stats()
        return new_root
    
    def _rotate_right(self, node: TreapNode) -> TreapNode:
        """Right rotation maintaining priorities"""
        if not node.left:
            return node
            
        new_root = node.left
        node.left = new_root.right
        new_root.right = node
        
        node.update_stats()
        new_root.update_stats()
        return new_root
    
    def find_overlapping(self, start: int, end: int) -> List[TreapInterval]:
        """Find all intervals overlapping with query range"""
        result = []
        self._find_overlapping(self.root, start, end, result)
        return result
    
    def _find_overlapping(self, node: Optional[TreapNode], start: int, end: int, result: List) -> None:
        """Recursive overlap search"""
        if not node:
            return
            
        # Check overlap with current node
        if node.start < end and node.end > start:
            result.append(TreapInterval(node.start, node.end, node.priority, node.data))
        
        # Search children
        if node.left and node.left.total_length > 0:
            self._find_overlapping(node.left, start, end, result)
        if node.right and node.right.total_length > 0:
            self._find_overlapping(node.right, start, end, result)
    
    def get_intervals(self) -> List[TreapInterval]:
        """Get all intervals in sorted order"""
        result = []
        self._inorder_traversal(self.root, result)
        return result
    
    def _inorder_traversal(self, node: Optional[TreapNode], result: List) -> None:
        """In-order traversal for sorted intervals"""
        if not node:
            return
        self._inorder_traversal(node.left, result)
        result.append(TreapInterval(node.start, node.end, node.priority, node.data))
        self._inorder_traversal(node.right, result)
    
    def print_tree(self) -> None:
        """Print tree structure showing priorities"""
        self._print_tree(self.root, "", "")
    
    def _print_tree(self, node: Optional[TreapNode], indent: str, prefix: str) -> None:
        """Print tree with priorities"""
        if not node:
            return
            
        self._print_tree(node.right, indent + "    ", "┌── ")
        print(f"{indent}{prefix}[{node.start},{node.end}) p={node.priority:.3f}")
        self._print_tree(node.left, indent + "    ", "└── ")


def demo_treap_randomized_balancing():
    """Demonstrate treap's probabilistic balancing"""
    print("🎲 Treap Randomized Balancing Demo")
    print("=" * 50)
    
    treap = RandomizedIntervalTreap()
    
    # Insert intervals in sorted order (worst case for standard BST)
    print("Inserting intervals in sorted order...")
    intervals = [(i*10, i*10 + 5, f"task_{i}") for i in range(10)]
    
    for start, end, data in intervals:
        treap.insert(start, end, data)
        
    print(f"Tree height: {treap.root.height} (vs {len(intervals)} for unbalanced BST)")
    print(f"Total length: {treap.root.total_length}")
    
    print("\nTree structure (priorities determine balance):")
    treap.print_tree()
    
    # Demonstrate random search
    print(f"\n🔍 Finding overlaps with [25, 35):")
    overlaps = treap.find_overlapping(25, 35)
    for interval in overlaps:
        print(f"  Overlap: [{interval.start},{interval.end}) priority={interval.priority:.3f}")


def demo_monte_carlo_optimization():
    """Monte Carlo optimization for interval allocation"""
    print("\n🎯 Monte Carlo Interval Allocation Optimization")
    print("=" * 50)
    
    def evaluate_allocation(intervals: List[Tuple[int, int]], capacity: int) -> float:
        """Evaluate allocation quality"""
        treap = RandomizedIntervalTreap()
        
        # Try to allocate all intervals
        allocated = 0
        total_value = 0
        
        for start, end in intervals:
            # Check if allocation is feasible (simplified)
            value = end - start  # Value = duration
            treap.insert(start, end, f"alloc_{allocated}")
            allocated += 1
            total_value += value
        
        # Penalty for fragmentation
        all_intervals = treap.get_intervals()
        if len(all_intervals) > 1:
            gaps = []
            sorted_intervals = sorted(all_intervals, key=lambda x: x.start)
            for i in range(len(sorted_intervals) - 1):
                gap = sorted_intervals[i+1].start - sorted_intervals[i].end
                if gap > 0:
                    gaps.append(gap)
            
            fragmentation_penalty = len(gaps) * 10  # Penalty for each gap
            total_value -= fragmentation_penalty
        
        return total_value
    
    # Generate random allocation scenarios
    best_allocation = None
    best_score = float('-inf')
    
    print("Running Monte Carlo optimization...")
    for iteration in range(1000):
        # Generate random interval allocation
        num_intervals = random.randint(5, 15)
        intervals = []
        
        for _ in range(num_intervals):
            start = random.randint(0, 90)
            duration = random.randint(5, 20)
            intervals.append((start, start + duration))
        
        # Evaluate allocation
        score = evaluate_allocation(intervals, 100)
        
        if score > best_score:
            best_score = score
            best_allocation = intervals
        
        if iteration % 200 == 0:
            print(f"  Iteration {iteration}: Best score = {best_score:.1f}")
    
    print(f"\n🏆 Best allocation found (score: {best_score:.1f}):")
    for i, (start, end) in enumerate(best_allocation):
        print(f"  Interval {i}: [{start}, {end}) length={end-start}")


def demo_randomized_load_balancing():
    """Randomized load balancing across multiple trees"""
    print("\n⚖️ Randomized Load Balancing Demo")
    print("=" * 50)
    
    # Multiple servers, each with their own treap
    num_servers = 4
    servers = [RandomizedIntervalTreap() for _ in range(num_servers)]
    server_loads = [0] * num_servers
    
    print(f"Load balancing across {num_servers} servers...")
    
    # Generate random tasks
    tasks = []
    for i in range(20):
        start = random.randint(0, 100)
        duration = random.randint(5, 25)
        tasks.append((start, start + duration, f"task_{i}"))
    
    # Randomized load balancing algorithms
    algorithms = {
        "Random": lambda: random.randint(0, num_servers - 1),
        "Power of Two": lambda: min(random.sample(range(num_servers), 2), 
                                   key=lambda s: server_loads[s]),
        "Weighted Random": lambda: random.choices(range(num_servers), 
                                                 weights=[1/(load+1) for load in server_loads])[0]
    }
    
    for alg_name, selector in algorithms.items():
        # Reset servers
        servers = [RandomizedIntervalTreap() for _ in range(num_servers)]
        server_loads = [0] * num_servers
        
        print(f"\n{alg_name} allocation:")
        
        for start, end, task_id in tasks:
            server_idx = selector()
            servers[server_idx].insert(start, end, task_id)
            server_loads[server_idx] += (end - start)
        
        # Print load distribution
        max_load = max(server_loads)
        min_load = min(server_loads)
        load_balance = max_load - min_load
        
        print(f"  Server loads: {server_loads}")
        print(f"  Load balance (max-min): {load_balance}")
        print(f"  Standard deviation: {(sum((x - sum(server_loads)/num_servers)**2 for x in server_loads)/num_servers)**0.5:.1f}")


def demo_probabilistic_scheduling():
    """Probabilistic scheduling with uncertain task durations"""
    print("\n📊 Probabilistic Scheduling Demo")
    print("=" * 50)
    
    class ProbabilisticTask:
        def __init__(self, name: str, expected_duration: float, variance: float):
            self.name = name
            self.expected_duration = expected_duration
            self.variance = variance
            
        def sample_duration(self) -> int:
            """Sample actual duration from distribution"""
            return max(1, int(random.gauss(self.expected_duration, self.variance**0.5)))
    
    # Create tasks with uncertain durations
    tasks = [
        ProbabilisticTask("Database Backup", 30, 25),
        ProbabilisticTask("Log Analysis", 15, 9),
        ProbabilisticTask("Report Generation", 20, 16),
        ProbabilisticTask("System Maintenance", 45, 100),
        ProbabilisticTask("Data Sync", 10, 4),
    ]
    
    print("Tasks with uncertain durations:")
    for task in tasks:
        print(f"  {task.name}: {task.expected_duration:.1f} ± {task.variance**0.5:.1f}")
    
    # Monte Carlo simulation of different scheduling strategies
    def simulate_schedule(strategy: str, num_simulations: int = 1000) -> dict:
        total_completion_times = []
        total_delays = []
        
        for sim in range(num_simulations):
            treap = RandomizedIntervalTreap()
            current_time = 0
            
            if strategy == "Random Order":
                task_order = random.sample(tasks, len(tasks))
            elif strategy == "Shortest Expected First":
                task_order = sorted(tasks, key=lambda t: t.expected_duration)
            else:  # "Longest Expected First"
                task_order = sorted(tasks, key=lambda t: t.expected_duration, reverse=True)
            
            for task in task_order:
                duration = task.sample_duration()
                treap.insert(current_time, current_time + duration, task.name)
                current_time += duration
            
            total_completion_times.append(current_time)
            
            # Calculate delay vs expected
            expected_total = sum(task.expected_duration for task in tasks)
            delay = current_time - expected_total
            total_delays.append(delay)
        
        return {
            'avg_completion': sum(total_completion_times) / num_simulations,
            'std_completion': (sum((x - sum(total_completion_times)/num_simulations)**2 
                                 for x in total_completion_times) / num_simulations)**0.5,
            'avg_delay': sum(total_delays) / num_simulations,
            'delay_variance': sum(d**2 for d in total_delays) / num_simulations
        }
    
    strategies = ["Random Order", "Shortest Expected First", "Longest Expected First"]
    
    print(f"\nMonte Carlo simulation results ({1000} simulations each):")
    for strategy in strategies:
        results = simulate_schedule(strategy)
        print(f"\n{strategy}:")
        print(f"  Avg completion time: {results['avg_completion']:.1f} ± {results['std_completion']:.1f}")
        print(f"  Avg delay from expected: {results['avg_delay']:.1f}")
        print(f"  Delay variance: {results['delay_variance']:.1f}")


def main():
    """Run all treap demonstrations"""
    print("🌳 Randomized Interval Tree (Treap) Demonstrations")
    print("Demonstrating probabilistic balancing and randomized algorithms")
    print("=" * 60)
    
    random.seed(42)  # For reproducible results
    
    demo_treap_randomized_balancing()
    demo_monte_carlo_optimization()
    demo_randomized_load_balancing()
    demo_probabilistic_scheduling()
    
    print("\n" + "=" * 60)
    print("✅ Treap demonstrations complete!")
    print("\n🎯 Key insights demonstrated:")
    print("  • Probabilistic balancing eliminates worst-case behavior")
    print("  • Monte Carlo optimization finds good solutions")
    print("  • Randomized load balancing improves fairness")
    print("  • Probabilistic scheduling handles uncertainty")
    print("  • Random priorities ensure O(log n) expected performance")


if __name__ == "__main__":
    main()
