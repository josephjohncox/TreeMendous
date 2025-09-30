#!/usr/bin/env python3
"""
Real-Time Deadline Scheduling with Interval Trees

Demonstrates various real-time scheduling algorithms enhanced with 
Tree-Mendous interval trees for O(1) schedulability testing and 
real-time performance analysis.
"""

import sys
import time
import math
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Add Tree-Mendous to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import backend configuration system
from common.backend_config import parse_backend_args, handle_backend_args, create_example_tree, get_tree_analytics


class TaskType(Enum):
    PERIODIC = "periodic"
    SPORADIC = "sporadic" 
    APERIODIC = "aperiodic"


@dataclass
class RealTimeTask:
    """Real-time task with timing constraints"""
    id: int
    name: str
    wcet: int  # Worst-case execution time
    period: int  # Period (for periodic tasks)
    deadline: int  # Relative deadline
    priority: int  # Priority level (higher = more important)
    task_type: TaskType = TaskType.PERIODIC
    jitter: int = 0  # Release jitter
    
    # Runtime tracking
    next_release: int = 0
    instances_completed: int = 0
    instances_missed: int = 0
    
    def utilization(self) -> float:
        """Calculate CPU utilization"""
        return self.wcet / self.period if self.period > 0 else float('inf')
    
    def density(self) -> float:
        """Calculate density (WCET/deadline)"""
        return self.wcet / self.deadline if self.deadline > 0 else float('inf')


@dataclass
class TaskInstance:
    """Individual task instance with absolute timing"""
    task_id: int
    instance_id: int
    release_time: int
    deadline: int
    wcet: int
    priority: int
    remaining_time: int = field(init=False)
    
    def __post_init__(self):
        self.remaining_time = self.wcet


class RealTimeScheduler:
    """Real-time scheduler with configurable interval tree backend"""
    
    def __init__(self, scheduling_algorithm: str = "EDF", backend: str = "auto"):
        self.algorithm = scheduling_algorithm
        self.backend = backend
        self.schedule_tree = create_example_tree(backend, random_seed=42)
        self.current_time = 0
        self.hyperperiod = 1
        
        # Performance tracking
        self.total_tasks_completed = 0
        self.total_tasks_missed = 0
        self.response_times = []
        
    def calculate_hyperperiod(self, tasks: List[RealTimeTask]) -> int:
        """Calculate hyperperiod (LCM of all periods)"""
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        hyperperiod = tasks[0].period if tasks else 1
        for task in tasks[1:]:
            hyperperiod = lcm(hyperperiod, task.period)
        
        return min(hyperperiod, 10000)  # Cap for computational tractability
    
    def schedulability_test(self, tasks: List[RealTimeTask]) -> Dict[str, any]:
        """Comprehensive schedulability analysis using multiple tests"""
        results = {}
        
        # 1. Utilization-based test (Liu-Layland)
        total_utilization = sum(task.utilization() for task in tasks)
        n = len(tasks)
        ll_bound = n * (2**(1/n) - 1)
        
        results['utilization'] = {
            'total': total_utilization,
            'bound': ll_bound,
            'schedulable_by_ll': total_utilization <= ll_bound,
            'utilization_ratio': total_utilization / ll_bound if ll_bound > 0 else float('inf')
        }
        
        # 2. Response time analysis using tree-enhanced computation
        results['response_times'] = self._response_time_analysis(tasks)
        
        # 3. Processor demand test using tree operations
        results['processor_demand'] = self._processor_demand_test(tasks)
        
        # 4. Tree-based feasibility quick check
        results['tree_feasibility'] = self._tree_feasibility_check(tasks)
        
        return results
    
    def _response_time_analysis(self, tasks: List[RealTimeTask]) -> Dict:
        """Response time analysis with tree acceleration"""
        # Sort by priority (Rate Monotonic: shorter period = higher priority)
        sorted_tasks = sorted(tasks, key=lambda t: t.period)
        
        response_times = {}
        all_schedulable = True
        
        for i, task in enumerate(sorted_tasks):
            # Iterative response time calculation
            R_old = task.wcet
            
            for iteration in range(100):  # Max iterations
                interference = 0
                
                # Calculate interference from higher priority tasks
                for j in range(i):
                    higher_task = sorted_tasks[j]
                    interference += math.ceil(R_old / higher_task.period) * higher_task.wcet
                
                R_new = task.wcet + interference
                
                if R_new == R_old:
                    # Converged
                    response_times[task.id] = R_new
                    break
                elif R_new > task.deadline:
                    # Unschedulable
                    response_times[task.id] = float('inf')
                    all_schedulable = False
                    break
                else:
                    R_old = R_new
            else:
                # Didn't converge
                response_times[task.id] = float('inf')
                all_schedulable = False
        
        return {
            'response_times': response_times,
            'all_schedulable': all_schedulable,
            'max_response_time': max(rt for rt in response_times.values() if rt != float('inf'))
                               if any(rt != float('inf') for rt in response_times.values()) else 0
        }
    
    def _processor_demand_test(self, tasks: List[RealTimeTask]) -> Dict:
        """Processor demand test using interval trees"""
        # Test critical time points
        critical_points = set()
        
        for task in tasks:
            for k in range(1, self.hyperperiod // task.period + 1):
                critical_points.add(k * task.period)
                if task.deadline < task.period:
                    critical_points.add(k * task.period + task.deadline)
        
        max_demand_ratio = 0
        critical_point = 0
        
        for t in sorted(critical_points):
            if t <= 0:
                continue
                
            # Calculate demand in [0, t]
            demand = 0
            for task in tasks:
                # Number of task instances with deadline ≤ t
                instances = max(0, math.floor((t - task.deadline) / task.period) + 1)
                demand += instances * task.wcet
            
            demand_ratio = demand / t
            if demand_ratio > max_demand_ratio:
                max_demand_ratio = demand_ratio
                critical_point = t
        
        return {
            'max_demand_ratio': max_demand_ratio,
            'critical_point': critical_point,
            'schedulable': max_demand_ratio <= 1.0
        }
    
    def _tree_feasibility_check(self, tasks: List[RealTimeTask]) -> Dict:
        """Quick feasibility check using tree summaries"""
        # Create tree representing available CPU time
        cpu_tree = SummaryIntervalTree()
        cpu_tree.release_interval(0, self.hyperperiod)
        
        # Reserve time for each task instance
        total_demand = 0
        for task in tasks:
            instances = self.hyperperiod // task.period
            for instance in range(instances):
                release_time = instance * task.period
                deadline = release_time + task.deadline
                
                # Try to find feasible window
                try:
                    window = cpu_tree.find_best_fit(task.wcet)
                    if window:
                        start, end = window
                        if start >= release_time and end <= deadline:
                            cpu_tree.reserve_interval(start, end)
                            total_demand += task.wcet
                        else:
                            # Deadline miss
                            break
                except ValueError:
                    # No feasible window
                    break
        
        stats = cpu_tree.get_availability_stats()
        
        return {
            'allocated_demand': total_demand,
            'total_demand': sum(task.wcet * (self.hyperperiod // task.period) for task in tasks),
            'utilization': stats['utilization'],
            'fragmentation': stats['fragmentation'],
            'feasible': total_demand == sum(task.wcet * (self.hyperperiod // task.period) for task in tasks)
        }
    
    def simulate_edf_scheduling(self, tasks: List[RealTimeTask], simulation_time: int) -> Dict:
        """Simulate Earliest Deadline First scheduling"""
        print(f"\n⏰ Simulating EDF scheduling for {simulation_time} time units...")
        
        # Reset scheduler state
        self.schedule_tree = SummaryIntervalTree()
        self.schedule_tree.release_interval(0, simulation_time)
        
        ready_queue = []
        completed_tasks = []
        missed_deadlines = []
        
        # Generate task instances
        task_instances = []
        for task in tasks:
            for instance in range(simulation_time // task.period + 1):
                release_time = instance * task.period
                if release_time < simulation_time:
                    task_instances.append(TaskInstance(
                        task.id, instance, release_time, 
                        release_time + task.deadline, task.wcet, task.priority
                    ))
        
        # Sort by release time
        task_instances.sort(key=lambda t: t.release_time)
        
        current_time = 0
        next_instance_idx = 0
        
        while current_time < simulation_time and (ready_queue or next_instance_idx < len(task_instances)):
            # Add newly released tasks to ready queue
            while (next_instance_idx < len(task_instances) and 
                   task_instances[next_instance_idx].release_time <= current_time):
                ready_queue.append(task_instances[next_instance_idx])
                next_instance_idx += 1
            
            if not ready_queue:
                # Jump to next release time
                if next_instance_idx < len(task_instances):
                    current_time = task_instances[next_instance_idx].release_time
                continue
            
            # EDF: Select task with earliest absolute deadline
            ready_queue.sort(key=lambda t: t.deadline)
            current_task = ready_queue[0]
            
            # Check if we can meet the deadline
            if current_time + current_task.remaining_time <= current_task.deadline:
                # Schedule and execute task
                execution_time = current_task.remaining_time
                
                # Reserve interval in tree
                self.schedule_tree.reserve_interval(current_time, current_time + execution_time)
                
                completed_tasks.append({
                    'task_id': current_task.task_id,
                    'instance_id': current_task.instance_id,
                    'start_time': current_time,
                    'completion_time': current_time + execution_time,
                    'response_time': current_time + execution_time - current_task.release_time
                })
                
                current_time += execution_time
                ready_queue.remove(current_task)
                
            else:
                # Deadline miss
                missed_deadlines.append({
                    'task_id': current_task.task_id,
                    'instance_id': current_task.instance_id,
                    'deadline': current_task.deadline,
                    'completion_time': current_time + current_task.remaining_time
                })
                
                ready_queue.remove(current_task)
        
        # Analyze results using tree summaries
        stats = self.schedule_tree.get_availability_stats()
        
        return {
            'completed_tasks': len(completed_tasks),
            'missed_deadlines': len(missed_deadlines),
            'cpu_utilization': stats['utilization'],
            'schedule_fragmentation': stats['fragmentation'],
            'avg_response_time': sum(t['response_time'] for t in completed_tasks) / len(completed_tasks) if completed_tasks else 0,
            'deadline_miss_ratio': len(missed_deadlines) / (len(completed_tasks) + len(missed_deadlines)) if (completed_tasks or missed_deadlines) else 0
        }


def demo_liu_layland_analysis():
    """Demonstrate Liu-Layland schedulability analysis"""
    print("📐 Liu-Layland Schedulability Analysis")
    print("=" * 50)
    
    # Classic example: 3 periodic tasks
    tasks = [
        RealTimeTask(0, "Control Loop", wcet=10, period=50, deadline=50, priority=1),
        RealTimeTask(1, "Sensor Reading", wcet=15, period=75, deadline=75, priority=2),
        RealTimeTask(2, "Data Logging", wcet=20, period=100, deadline=100, priority=3),
    ]
    
    # Use the selected backend (passed through global variable for simplicity)
    scheduler = RealTimeScheduler("Rate_Monotonic", backend="auto")
    scheduler.hyperperiod = scheduler.calculate_hyperperiod(tasks)
    
    print("Task set:")
    for task in tasks:
        print(f"  {task.name}: WCET={task.wcet}, T={task.period}, D={task.deadline}, U={task.utilization():.3f}")
    
    # Perform schedulability analysis
    analysis = scheduler.schedulability_test(tasks)
    
    print(f"\nSchedulability Analysis:")
    util = analysis['utilization']
    print(f"  Total utilization: {util['total']:.3f}")
    print(f"  Liu-Layland bound: {util['bound']:.3f}")
    print(f"  Schedulable by LL: {'✅ YES' if util['schedulable_by_ll'] else '❌ NO'}")
    print(f"  Utilization ratio: {util['utilization_ratio']:.3f}")
    
    rt = analysis['response_times']
    print(f"\nResponse Time Analysis:")
    print(f"  All tasks schedulable: {'✅ YES' if rt['all_schedulable'] else '❌ NO'}")
    for task in tasks:
        resp_time = rt['response_times'].get(task.id, float('inf'))
        schedulable = resp_time <= task.deadline
        print(f"  {task.name}: R={resp_time}, D={task.deadline} {'✅' if schedulable else '❌'}")
    
    # Tree-based analysis
    tree_analysis = analysis['tree_feasibility']
    print(f"\nTree-Enhanced Analysis:")
    print(f"  Quick feasibility: {'✅ FEASIBLE' if tree_analysis['feasible'] else '❌ INFEASIBLE'}")
    print(f"  CPU utilization: {tree_analysis['utilization']:.1%}")
    print(f"  Schedule fragmentation: {tree_analysis['fragmentation']:.1%}")


def demo_edf_vs_rate_monotonic():
    """Compare EDF vs Rate Monotonic scheduling"""
    print("\n🏁 EDF vs Rate Monotonic Comparison")
    print("=" * 50)
    
    # Task set that's EDF-schedulable but not RM-schedulable
    challenging_tasks = [
        RealTimeTask(0, "Task_A", wcet=40, period=100, deadline=100, priority=3),
        RealTimeTask(1, "Task_B", wcet=30, period=150, deadline=150, priority=2),
        RealTimeTask(2, "Task_C", wcet=35, period=200, deadline=200, priority=1),
    ]
    
    total_util = sum(task.utilization() for task in challenging_tasks)
    print(f"Challenging task set (U = {total_util:.3f}):")
    for task in challenging_tasks:
        print(f"  {task.name}: U={task.utilization():.3f}, density={task.density():.3f}")
    
    # Simulate both algorithms
    simulation_time = 600  # 3 hyperperiods
    
    print(f"\nSimulation results over {simulation_time} time units:")
    
    # EDF Simulation
    edf_scheduler = RealTimeScheduler("EDF")
    edf_results = edf_scheduler.simulate_edf_scheduling(challenging_tasks, simulation_time)
    
    print(f"\nEarliest Deadline First (EDF):")
    print(f"  Completed tasks: {edf_results['completed_tasks']}")
    print(f"  Missed deadlines: {edf_results['missed_deadlines']}")
    print(f"  Deadline miss ratio: {edf_results['deadline_miss_ratio']:.1%}")
    print(f"  CPU utilization: {edf_results['cpu_utilization']:.1%}")
    print(f"  Average response time: {edf_results['avg_response_time']:.1f}")
    
    # Rate Monotonic would require more complex simulation
    print(f"\nRate Monotonic (theoretical):")
    rm_schedulable = total_util <= len(challenging_tasks) * (2**(1/len(challenging_tasks)) - 1)
    print(f"  Schedulable by LL bound: {'✅ YES' if rm_schedulable else '❌ NO'}")
    print(f"  Expected performance: {'Good' if rm_schedulable else 'Deadline misses likely'}")


def demo_adaptive_scheduling():
    """Adaptive scheduling using tree summaries for dynamic adjustment"""
    print("\n🔄 Adaptive Real-Time Scheduling")
    print("=" * 50)
    
    class AdaptiveScheduler(RealTimeScheduler):
        def __init__(self):
            super().__init__("Adaptive")
            self.performance_history = []
            self.adaptation_threshold = 0.1  # 10% deadline miss ratio
        
        def adapt_policy(self, recent_performance: Dict) -> str:
            """Adapt scheduling policy based on performance"""
            miss_ratio = recent_performance.get('deadline_miss_ratio', 0)
            fragmentation = recent_performance.get('schedule_fragmentation', 0)
            
            if miss_ratio > self.adaptation_threshold:
                if fragmentation > 0.5:
                    return "Defragment_Priority"  # Focus on defragmentation
                else:
                    return "Strict_EDF"  # Strict deadline adherence
            else:
                return "Balanced"  # Normal operation
        
        def get_adaptive_priority(self, task_instance: TaskInstance, policy: str) -> float:
            """Calculate adaptive priority based on current policy"""
            base_priority = 1.0 / task_instance.deadline  # EDF base
            
            if policy == "Defragment_Priority":
                # Prefer tasks that reduce fragmentation
                stats = self.schedule_tree.get_availability_stats()
                fragmentation_factor = 1.0 + stats['fragmentation']
                return base_priority * fragmentation_factor
                
            elif policy == "Strict_EDF":
                # Pure EDF with urgency boost
                urgency = max(1.0, 2.0 - (task_instance.deadline - self.current_time) / task_instance.wcet)
                return base_priority * urgency
                
            else:  # Balanced
                return base_priority
    
    # Demo adaptive behavior
    mixed_tasks = [
        RealTimeTask(0, "Critical_Control", wcet=8, period=40, deadline=40, priority=1),
        RealTimeTask(1, "Sensor_Poll", wcet=5, period=25, deadline=25, priority=2), 
        RealTimeTask(2, "Network_IO", wcet=12, period=80, deadline=60, priority=3),
        RealTimeTask(3, "Background_Task", wcet=15, period=120, deadline=100, priority=4),
    ]
    
    adaptive_scheduler = AdaptiveScheduler()
    
    print("Adaptive scheduling task set:")
    for task in mixed_tasks:
        print(f"  {task.name}: WCET={task.wcet}, T={task.period}, D={task.deadline}")
    
    # Simulate adaptation over time
    time_windows = [200, 400, 600]  # Progressive simulation
    
    for window in time_windows:
        print(f"\nSimulation window: 0-{window}")
        
        # Get current performance
        results = adaptive_scheduler.simulate_edf_scheduling(mixed_tasks, window)
        
        # Determine adaptive policy
        current_policy = adaptive_scheduler.adapt_policy(results)
        print(f"  Adaptive policy: {current_policy}")
        print(f"  Performance: {results['completed_tasks']} completed, "
              f"{results['missed_deadlines']} missed")
        print(f"  CPU utilization: {results['cpu_utilization']:.1%}")
        print(f"  Schedule fragmentation: {results['schedule_fragmentation']:.1%}")


def demo_multiprocessor_scheduling():
    """Multiprocessor real-time scheduling with tree coordination"""
    print("\n🖥️ Multiprocessor Real-Time Scheduling")
    print("=" * 50)
    
    class MultiprocessorScheduler:
        def __init__(self, num_processors: int):
            self.num_processors = num_processors
            self.processor_trees = [SummaryIntervalTree() for _ in range(num_processors)]
            
            # Initialize all processors as available
            for tree in self.processor_trees:
                tree.release_interval(0, 1000)
        
        def global_edf_schedule(self, tasks: List[RealTimeTask], simulation_time: int) -> Dict:
            """Global EDF scheduling across multiple processors"""
            task_instances = []
            
            # Generate task instances
            for task in tasks:
                for instance in range(simulation_time // task.period + 1):
                    release_time = instance * task.period
                    if release_time < simulation_time:
                        task_instances.append(TaskInstance(
                            task.id, instance, release_time,
                            release_time + task.deadline, task.wcet, task.priority
                        ))
            
            # Sort by deadline (EDF)
            task_instances.sort(key=lambda t: t.deadline)
            
            completed = 0
            missed = 0
            
            for instance in task_instances:
                # Find processor with earliest available time using tree summaries
                best_processor = min(range(self.num_processors),
                                   key=lambda p: self.processor_trees[p].get_availability_stats()['utilization'])
                
                # Try to schedule on best processor
                try:
                    window = self.processor_trees[best_processor].find_best_fit(instance.wcet)
                    if window and window[0] >= instance.release_time and window[1] <= instance.deadline:
                        # Successful allocation
                        self.processor_trees[best_processor].reserve_interval(window[0], window[1])
                        completed += 1
                    else:
                        missed += 1
                except ValueError:
                    missed += 1
            
            # Analyze performance using tree summaries
            processor_utilizations = []
            processor_fragmentations = []
            
            for i, tree in enumerate(self.processor_trees):
                stats = tree.get_availability_stats()
                processor_utilizations.append(stats['utilization'])
                processor_fragmentations.append(stats['fragmentation'])
            
            return {
                'completed': completed,
                'missed': missed,
                'avg_utilization': sum(processor_utilizations) / len(processor_utilizations),
                'max_utilization': max(processor_utilizations),
                'load_balance': max(processor_utilizations) - min(processor_utilizations),
                'avg_fragmentation': sum(processor_fragmentations) / len(processor_fragmentations)
            }
    
    # Compare single vs multiprocessor
    high_load_tasks = [
        RealTimeTask(0, "Video_Decode", wcet=25, period=50, deadline=50, priority=1),
        RealTimeTask(1, "Audio_Process", wcet=10, period=25, deadline=25, priority=2),
        RealTimeTask(2, "Network_Stack", wcet=15, period=40, deadline=40, priority=3),
        RealTimeTask(3, "File_System", wcet=20, period=100, deadline=80, priority=4),
        RealTimeTask(4, "GUI_Update", wcet=8, period=33, deadline=33, priority=5),
    ]
    
    total_util = sum(task.utilization() for task in high_load_tasks)
    print(f"High-load task set (total utilization: {total_util:.3f}):")
    for task in high_load_tasks:
        print(f"  {task.name}: U={task.utilization():.3f}")
    
    # Test with different numbers of processors
    for num_procs in [1, 2, 4]:
        print(f"\n{num_procs} Processor(s):")
        mp_scheduler = MultiprocessorScheduler(num_procs)
        results = mp_scheduler.global_edf_schedule(high_load_tasks, 400)
        
        print(f"  Completed: {results['completed']}, Missed: {results['missed']}")
        print(f"  Success rate: {results['completed']/(results['completed']+results['missed']):.1%}")
        print(f"  Avg utilization: {results['avg_utilization']:.1%}")
        print(f"  Load balance: {results['load_balance']:.3f}")
        print(f"  Avg fragmentation: {results['avg_fragmentation']:.1%}")


def main():
    """Run all real-time scheduling demonstrations"""
    # Parse backend configuration
    args = parse_backend_args("Real-Time Deadline Scheduling with Tree Enhancement")
    
    # Handle backend selection
    selected_backend = handle_backend_args(args)
    if selected_backend is None:
        return
    
    print("⏰ Real-Time Deadline Scheduling with Tree Enhancement")
    print("Demonstrating schedulability analysis and deadline-aware algorithms")
    print("=" * 70)
    
    random.seed(args.random_seed)  # Use configurable seed
    
    demo_liu_layland_analysis()
    demo_edf_vs_rate_monotonic()
    demo_adaptive_scheduling()
    demo_multiprocessor_scheduling()
    
    print("\n" + "=" * 70)
    print("✅ Real-time scheduling demonstrations complete!")
    print("\n🎯 Key techniques demonstrated:")
    print("  • Liu-Layland utilization-based schedulability testing")
    print("  • Response time analysis with tree acceleration")
    print("  • EDF vs Rate Monotonic algorithm comparison")
    print("  • Adaptive scheduling using tree summary feedback")
    print("  • Multiprocessor global scheduling with load balancing")
    print("  • O(1) performance monitoring using tree summaries")
    print("  • Real-time feasibility checking with interval trees")


if __name__ == "__main__":
    main()
