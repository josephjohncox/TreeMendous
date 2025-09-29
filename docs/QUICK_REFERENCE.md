# 🚀 Tree-Mendous: Quick Reference Guide

## One-Minute Start

```python
from treemendous.basic.boundary import IntervalManager

# Create manager and allocate space
manager = IntervalManager()
manager.release_interval(0, 1000)      # Mark [0,1000) as available
manager.reserve_interval(200, 400)     # Allocate [200,400)

# Find available space
result = manager.find_interval(0, 100) # Find 100 units starting at 0
print(f"Allocated: [{result.start}, {result.end})")

# Check status
print(f"Available: {manager.get_total_available_length()} units")
```

## 🎯 Choose Your Implementation

| Need | Use | Import |
|------|-----|--------|
| **Simple & Fast** | `IntervalManager` | `from treemendous.basic.boundary import IntervalManager` |
| **Analytics** | `SummaryIntervalTree` | `from treemendous.basic.summary import SummaryIntervalTree` |
| **Random/Fair** | `IntervalTreap` | `from treemendous.basic.treap import IntervalTreap` |
| **High Performance** | `BoundarySummaryManager` | `from treemendous.basic.boundary_summary import BoundarySummaryManager` |
| **C++ Speed** | Any with `cpp_` prefix | `from treemendous.backend_manager import create_interval_tree` |

## 🔧 Core Operations

### Basic Operations (All Implementations)

```python
# Mark space as available
manager.release_interval(start, end)

# Mark space as occupied  
manager.reserve_interval(start, end)

# Find available space
result = manager.find_interval(point, length)
# Returns: IntervalResult(start, end, length) or None

# Get all available intervals
intervals = manager.get_intervals()
# Returns: List[IntervalResult]

# Get total available space
total = manager.get_total_available_length()
# Returns: int
```

### Enhanced Operations (Summary/BoundarySummary)

```python
# Smart allocation
best = manager.find_best_fit(size)         # Minimize waste
largest = manager.find_largest_available() # Maximum capacity

# Analytics
summary = manager.get_summary()
print(f"Fragmentation: {summary.fragmentation_index:.2f}")
print(f"Utilization: {summary.utilization:.1%}")
```

### Randomized Operations (Treap Only)

```python
# Random sampling
sample = treap.sample_random_interval()

# Tree surgery
left, right = treap.split(position)
merged = left.merge_treap(right)

# Probabilistic properties
valid = treap.verify_treap_properties()
```

## 📊 Return Types

All methods return standardized objects:

```python
@dataclass(frozen=True)
class IntervalResult:
    start: int      # Start position
    end: int        # End position (exclusive)
    length: int     # Computed: end - start
    data: Any       # Optional metadata
```

## ⚡ Performance Quick Facts

| Implementation | Insert | Query | Memory/Interval | Best For |
|----------------|--------|-------|-----------------|----------|
| **Boundary** | O(log n) | O(log n) | 16 bytes | General use |
| **Summary** | O(log n) | O(log n) | 80 bytes | Analytics |
| **Treap** | O(log n) | O(log n) | 48 bytes | Dynamic load |
| **BoundarySummary** | O(log n) | O(1)* | 16 + cache | Query-heavy |

*\* When cached*

## 🛠️ Common Patterns

### Pattern 1: Smart Backend Selection

```python
from treemendous.backend_manager import create_interval_tree

# Automatic selection based on available backends
tree = create_interval_tree("auto")

# Force specific implementation
tree = create_interval_tree("cpp_boundary")  # C++ for speed
tree = create_interval_tree("py_treap")      # Python treap
```

### Pattern 2: Error-Safe Operations

```python
def safe_allocate(manager, start, length):
    try:
        result = manager.find_interval(start, length)
        if result:
            manager.reserve_interval(result.start, result.end)
            return result
        else:
            raise ValueError(f"No space for {length} units")
    except Exception as e:
        print(f"Allocation failed: {e}")
        return None
```

### Pattern 3: Performance Monitoring

```python
def monitored_operation(manager, operation, *args):
    start_time = time.perf_counter()
    result = getattr(manager, operation)(*args)
    duration = time.perf_counter() - start_time
    
    print(f"{operation}: {duration*1000:.2f}ms")
    return result
```

## 🔍 Debugging Tips

### View Current State

```python
# All implementations
for interval in manager.get_intervals():
    print(f"[{interval.start}, {interval.end}) - {interval.length} units")

# With tree structure (Summary/Treap)
if hasattr(manager, 'print_tree'):
    manager.print_tree()
```

### Validate Correctness

```python
# Check total length consistency
intervals = manager.get_intervals()
calculated = sum(interval.length for interval in intervals)
reported = manager.get_total_available_length()
assert calculated == reported, f"Length mismatch: {calculated} vs {reported}"
```

### Performance Analysis

```python
# For BoundarySummaryManager
perf = manager.get_performance_stats()
print(f"Operations: {perf.operation_count}")
print(f"Cache hits: {perf.cache_hits}")
print(f"Hit rate: {perf.cache_hit_rate:.1%}")

# For Treap
valid = treap.verify_treap_properties()
print(f"Tree properties valid: {valid}")
```

## 🎯 Real-World Examples

### Memory Pool Manager

```python
heap = IntervalManager()
heap.release_interval(0, 1024*1024)  # 1MB heap

def malloc(size):
    allocation = heap.find_interval(0, size)
    if allocation:
        heap.reserve_interval(allocation.start, allocation.end)
        return allocation.start  # Return pointer
    return None  # Out of memory

def free(ptr, size):
    heap.release_interval(ptr, ptr + size)
```

### CPU Scheduler

```python
scheduler = SummaryIntervalTree()
scheduler.release_interval(0, 1000)  # 1000ms time slice

# Schedule high-priority task
task = scheduler.find_interval(0, 100)  # Must start early
scheduler.reserve_interval(task.start, task.end)

# Check system load
summary = scheduler.get_tree_summary()
cpu_usage = 1.0 - (summary.total_free_length / 1000)
print(f"CPU usage: {cpu_usage:.1%}")
```

### Load Balancer

```python
balancer = IntervalTreap(random_seed=42)
balancer.release_interval(0, 1000)  # Total capacity

# Fair server allocation
servers = []
for i in range(10):
    allocation = balancer.sample_random_interval()
    if allocation:
        size = min(50, allocation.length)
        balancer.reserve_interval(allocation.start, allocation.start + size)
        servers.append(allocation.start)

print(f"Server distribution: {sorted(servers)}")
```

## 🔗 Quick Links

- **Full Documentation**: [INTERVAL_TREE_VISUALIZATION.md](INTERVAL_TREE_VISUALIZATION.md)
- **Architecture Deep Dive**: [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md)
- **Interactive Examples**: `examples/visualizations/`
- **Test Suite**: `just test`
- **Performance Benchmark**: `just benchmark`

---

*Get started in seconds, scale to millions of operations* ⚡
