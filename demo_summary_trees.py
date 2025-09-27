#!/usr/bin/env python3
"""
Demonstration of Enhanced Summary Interval Trees

This script showcases the new summary-enhanced interval trees that maintain
aggregate statistics for efficient scheduling queries. Perfect for resource
allocation, meeting scheduling, and capacity management scenarios.
"""

from treemendous.basic.summary import SummaryIntervalTree, TreeSummary


def print_separator(title: str) -> None:
    """Print a formatted section separator"""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_stats(tree: SummaryIntervalTree, context: str = "") -> None:
    """Print comprehensive tree statistics"""
    stats = tree.get_availability_stats()
    summary = tree.get_tree_summary()
    
    print(f"\n📊 {context} Statistics:")
    print(f"   Total Free Space: {stats['total_free']:,} units")
    print(f"   Total Occupied: {stats['total_occupied']:,} units") 
    print(f"   Free Chunks: {stats['free_chunks']}")
    print(f"   Largest Chunk: {stats['largest_chunk']:,} units")
    print(f"   Average Chunk: {stats['avg_chunk_size']:.1f} units")
    print(f"   Utilization: {stats['utilization']:.1%}")
    print(f"   Fragmentation: {stats['fragmentation']:.1%}")
    
    if summary.earliest_free_start is not None and summary.latest_free_end is not None:
        print(f"   Free Range: [{summary.earliest_free_start:,}, {summary.latest_free_end:,}]")


def demo_meeting_scheduler() -> None:
    """Demo: Meeting room scheduling system"""
    print_separator("Meeting Room Scheduler")
    
    tree = SummaryIntervalTree()
    
    # Initialize with a full business day (9 AM to 6 PM = 9 hours = 32400 seconds)
    BUSINESS_DAY_START = 9 * 3600   # 9 AM in seconds
    BUSINESS_DAY_END = 18 * 3600    # 6 PM in seconds
    
    tree.release_interval(BUSINESS_DAY_START, BUSINESS_DAY_END)
    print(f"🏢 Business day available: 9 AM - 6 PM")
    print_stats(tree, "Initial")
    
    # Schedule some meetings
    meetings = [
        (10 * 3600, 11 * 3600, "Team Standup"),      # 10-11 AM
        (11.5 * 3600, 12.5 * 3600, "Client Call"),  # 11:30 AM - 12:30 PM  
        (14 * 3600, 15.5 * 3600, "Planning Meeting"), # 2-3:30 PM
        (16 * 3600, 17 * 3600, "1-on-1 Review"),    # 4-5 PM
    ]
    
    print(f"\n📅 Scheduling {len(meetings)} meetings:")
    for start, end, name in meetings:
        tree.reserve_interval(int(start), int(end))
        start_hour = start / 3600
        end_hour = end / 3600
        print(f"   ✓ {name}: {start_hour:.1f}h - {end_hour:.1f}h")
    
    print_stats(tree, "After Scheduling")
    
    # Query for available slots
    print(f"\n🔍 Finding Available Slots:")
    
    # Find 1-hour slot
    one_hour = tree.find_best_fit(3600)  # 1 hour
    if one_hour:
        start_h = (one_hour[0] - BUSINESS_DAY_START) / 3600 + 9
        end_h = (one_hour[1] - BUSINESS_DAY_START) / 3600 + 9
        print(f"   • Best 1-hour slot: {start_h:.1f}h - {end_h:.1f}h")
    
    # Find 30-minute slot  
    thirty_min = tree.find_best_fit(1800)  # 30 minutes
    if thirty_min:
        start_h = (thirty_min[0] - BUSINESS_DAY_START) / 3600 + 9
        end_h = (thirty_min[1] - BUSINESS_DAY_START) / 3600 + 9
        print(f"   • Best 30-min slot: {start_h:.1f}h - {end_h:.1f}h")
    
    # Find largest available block
    largest = tree.find_largest_available()
    if largest:
        duration = (largest[1] - largest[0]) / 3600
        start_h = (largest[0] - BUSINESS_DAY_START) / 3600 + 9
        print(f"   • Largest block: {duration:.1f}h starting at {start_h:.1f}h")


def demo_memory_allocator() -> None:
    """Demo: Memory allocation system"""
    print_separator("Memory Allocator")
    
    tree = SummaryIntervalTree()
    
    # Initialize with 64MB heap
    HEAP_SIZE = 64 * 1024 * 1024  # 64MB
    tree.release_interval(0, HEAP_SIZE)
    
    print(f"💾 Memory heap: {HEAP_SIZE // (1024*1024)} MB available")
    print_stats(tree, "Initial Heap")
    
    # Allocate various memory blocks
    allocations = [
        (2 * 1024 * 1024, "Image Buffer"),     # 2MB
        (8 * 1024 * 1024, "Video Frame"),     # 8MB  
        (1024 * 1024, "Audio Buffer"),        # 1MB
        (4 * 1024 * 1024, "Texture Cache"),   # 4MB
        (512 * 1024, "Network Buffer"),       # 512KB
    ]
    
    print(f"\n🧠 Allocating memory blocks:")
    allocated_blocks = []
    
    for size, name in allocations:
        # Find best fit allocation
        slot = tree.find_best_fit(size, prefer_early=True)
        if slot:
            start, end = slot[0], slot[0] + size
            tree.reserve_interval(start, end)
            allocated_blocks.append((start, end, name))
            print(f"   ✓ {name}: {size // (1024*1024) if size >= 1024*1024 else size // 1024}{'MB' if size >= 1024*1024 else 'KB'} @ 0x{start:08x}")
        else:
            print(f"   ✗ {name}: Not enough contiguous space")
    
    print_stats(tree, "After Allocations")
    
    # Show memory fragmentation analysis
    print(f"\n🔍 Memory Analysis:")
    largest = tree.find_largest_available()
    if largest:
        largest_size = largest[1] - largest[0]
        print(f"   • Largest free block: {largest_size // (1024*1024)} MB")
    
    stats = tree.get_availability_stats()
    print(f"   • Fragmentation index: {stats['fragmentation']:.2f}")
    print(f"   • Memory efficiency: {(1-stats['fragmentation']):.1%}")
    
    # Free some blocks to demonstrate defragmentation
    print(f"\n🗑️  Freeing some allocations:")
    for start, end, name in allocated_blocks[1::2]:  # Free every other block
        tree.release_interval(start, end)
        print(f"   ✓ Freed {name}")
    
    print_stats(tree, "After Partial Deallocation")


def demo_server_capacity() -> None:
    """Demo: Server capacity management"""
    print_separator("Server Capacity Management")
    
    tree = SummaryIntervalTree()
    
    # Server has 1000 capacity units for the day
    DAY_DURATION = 24 * 3600  # 24 hours
    MAX_CAPACITY = 1000
    
    tree.release_interval(0, DAY_DURATION * MAX_CAPACITY)  # Time × Capacity
    
    print(f"🖥️  Server capacity: {MAX_CAPACITY} units × 24 hours")
    print_stats(tree, "Initial Capacity")
    
    # Reserve capacity for various workloads
    workloads = [
        (2*3600, 6*3600, 200, "Batch Processing"),     # 2-6 AM, 200 units
        (8*3600, 17*3600, 150, "Web Traffic"),        # 8 AM - 5 PM, 150 units  
        (12*3600, 13*3600, 300, "Peak Load"),         # 12-1 PM, 300 units (additional)
        (19*3600, 23*3600, 100, "Background Tasks"),  # 7-11 PM, 100 units
    ]
    
    print(f"\n⚡ Reserving server capacity:")
    for start_time, end_time, capacity, name in workloads:
        # Convert to capacity-time units
        duration = end_time - start_time
        capacity_units = duration * capacity
        
        # Find slot in capacity-time space
        slot = tree.find_best_fit(capacity_units)
        if slot:
            tree.reserve_interval(slot[0], slot[0] + capacity_units)
            start_h = start_time // 3600
            end_h = end_time // 3600
            print(f"   ✓ {name}: {capacity} units, {start_h}h-{end_h}h")
        else:
            print(f"   ✗ {name}: Insufficient capacity")
    
    print_stats(tree, "After Reservations")
    
    # Query available capacity
    print(f"\n🔍 Capacity Analysis:")
    
    # Find largest available capacity block
    largest = tree.find_largest_available()
    if largest:
        largest_capacity_time = largest[1] - largest[0]
        # Assume 4-hour block for estimation
        estimated_capacity = largest_capacity_time // (4 * 3600)
        print(f"   • Largest available: ~{estimated_capacity} units for 4h block")
    
    # Calculate utilization metrics
    stats = tree.get_availability_stats()
    utilization = stats['utilization']
    print(f"   • Overall utilization: {utilization:.1%}")
    print(f"   • Capacity fragmentation: {stats['fragmentation']:.1%}")


def demo_tree_summary_features() -> None:
    """Demo: TreeSummary data structure features"""
    print_separator("TreeSummary Data Structure")
    
    print("🌳 TreeSummary provides rich aggregate statistics:")
    print("   • total_free_length: Sum of all free space")
    print("   • total_occupied_length: Sum of all occupied space") 
    print("   • contiguous_count: Number of separate free intervals")
    print("   • largest_free_length: Size of largest free block")
    print("   • free_density: Ratio of free to total space")
    print("   • fragmentation: 1 - (largest_free / total_free)")
    
    # Create sample tree
    tree = SummaryIntervalTree()
    tree.release_interval(0, 1000)
    
    # Create fragmentation
    tree.reserve_interval(100, 200)  
    tree.reserve_interval(300, 400)
    tree.reserve_interval(600, 700)
    
    summary = tree.get_tree_summary()
    print(f"\n📈 Example Summary (range [0,1000] with 3 reserved blocks):")
    print(f"   • Total free: {summary.total_free_length}")
    print(f"   • Total occupied: {summary.total_occupied_length}")
    print(f"   • Free chunks: {summary.contiguous_count}")
    print(f"   • Largest chunk: {summary.largest_free_length}")
    print(f"   • Average chunk: {summary.avg_free_length:.1f}")
    print(f"   • Free density: {summary.free_density:.2f}")
    print(f"   • Bounds: [{summary.earliest_free_start}, {summary.latest_free_end}]")
    
    # Show merge functionality
    left_summary = TreeSummary.from_interval(0, 100)
    right_summary = TreeSummary.from_interval(200, 300)
    merged = TreeSummary.merge(left_summary, right_summary)
    
    print(f"\n🔄 TreeSummary Merging Example:")
    print(f"   Left interval [0, 100]:  free_length={left_summary.total_free_length}")
    print(f"   Right interval [200, 300]: free_length={right_summary.total_free_length}")
    print(f"   Merged result: free_length={merged.total_free_length}, chunks={merged.contiguous_count}")


def main() -> None:
    """Run all demonstrations"""
    print("🌟 Tree-Mendous: Enhanced Interval Trees with Summary Statistics")
    print("   Efficient scheduling through aggregate tree summaries")
    
    # Run all demos
    demo_meeting_scheduler()
    demo_memory_allocator() 
    demo_server_capacity()
    demo_tree_summary_features()
    
    print_separator("Summary")
    print("✅ Summary-enhanced trees provide:")
    print("   • O(1) aggregate statistics at every node")
    print("   • Fast best-fit queries using summary pruning")
    print("   • Comprehensive fragmentation analysis")
    print("   • Real-time utilization monitoring")
    print("   • Efficient resource allocation decisions")
    print("\n🚀 Perfect for scheduling systems, memory allocators,")
    print("   capacity management, and resource optimization!")


if __name__ == "__main__":
    main()
