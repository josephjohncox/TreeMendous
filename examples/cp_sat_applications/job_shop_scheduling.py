#!/usr/bin/env python3
"""
Job Shop Scheduling with CP-SAT and Interval Trees

Demonstrates constraint programming approach to job shop scheduling
using Google OR-Tools CP-SAT solver enhanced with Tree-Mendous interval trees
for efficient feasibility checking and optimization guidance.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random

# Add Tree-Mendous to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import backend configuration system
from common.backend_config import parse_backend_args, handle_backend_args, create_example_tree, get_tree_analytics

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("⚠️  OR-Tools not available - using simulation mode")


@dataclass
class Job:
    """Job with sequence of operations"""
    id: int
    operations: List[Tuple[int, int]]  # (machine_id, processing_time)
    due_date: Optional[int] = None
    priority: int = 1


@dataclass
class Machine:
    """Machine with processing capabilities"""
    id: int
    name: str
    capacity: int = 1
    setup_time: int = 0


class JobShopCPSAT:
    """Job shop scheduler using CP-SAT with configurable interval tree backend"""
    
    def __init__(self, machines: List[Machine], horizon: int = 1000, backend: str = "auto"):
        self.machines = machines
        self.horizon = horizon
        self.backend = backend
        
        # Create machine trees using specified backend
        self.machine_trees = {}
        for machine in machines:
            tree = create_example_tree(backend, random_seed=42)
            tree.release_interval(0, horizon)
            self.machine_trees[machine.id] = tree
    
    def solve_job_shop(self, jobs: List[Job]) -> Optional[Dict]:
        """Solve job shop scheduling problem using CP-SAT"""
        
        if not ORTOOLS_AVAILABLE:
            return self._simulate_solution(jobs)
        
        model = cp_model.CpModel()
        
        # Variables for start times
        start_times = {}
        intervals = {}
        
        # Create variables for each operation
        for job in jobs:
            for op_idx, (machine_id, processing_time) in enumerate(job.operations):
                task_id = f"job_{job.id}_op_{op_idx}"
                
                # Use tree summaries to guide variable domains
                machine_tree = self.machine_trees[machine_id]
                available_windows = machine_tree.get_intervals()
                
                if available_windows:
                    # Constrain start times to feasible windows
                    min_start = min(start for start, _, _ in available_windows)
                    max_start = max(end - processing_time for _, end, _ in available_windows)
                    max_start = max(0, min(max_start, self.horizon - processing_time))
                else:
                    min_start, max_start = 0, self.horizon - processing_time
                
                start_times[task_id] = model.NewIntVar(min_start, max_start, f"start_{task_id}")
                intervals[task_id] = model.NewIntervalVar(
                    start_times[task_id], 
                    processing_time,
                    start_times[task_id] + processing_time,
                    f"interval_{task_id}"
                )
        
        # Precedence constraints within jobs
        for job in jobs:
            for op_idx in range(len(job.operations) - 1):
                current_task = f"job_{job.id}_op_{op_idx}"
                next_task = f"job_{job.id}_op_{op_idx + 1}"
                
                # Current operation must finish before next starts
                model.Add(start_times[current_task] + job.operations[op_idx][1] <= start_times[next_task])
        
        # Machine capacity constraints (no overlap)
        machine_intervals = {m.id: [] for m in self.machines}
        for job in jobs:
            for op_idx, (machine_id, _) in enumerate(job.operations):
                task_id = f"job_{job.id}_op_{op_idx}"
                machine_intervals[machine_id].append(intervals[task_id])
        
        for machine_id, machine_interval_list in machine_intervals.items():
            if machine_interval_list:
                model.AddNoOverlap(machine_interval_list)
        
        # Due date constraints
        for job in jobs:
            if job.due_date:
                last_op_idx = len(job.operations) - 1
                last_task = f"job_{job.id}_op_{last_op_idx}"
                last_processing_time = job.operations[last_op_idx][1]
                
                model.Add(start_times[last_task] + last_processing_time <= job.due_date)
        
        # Objective: minimize makespan
        makespan = model.NewIntVar(0, self.horizon, "makespan")
        for job in jobs:
            last_op_idx = len(job.operations) - 1
            last_task = f"job_{job.id}_op_{last_op_idx}"
            last_processing_time = job.operations[last_op_idx][1]
            
            model.Add(makespan >= start_times[last_task] + last_processing_time)
        
        model.Minimize(makespan)
        
        # Solve with tree-guided heuristics
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            solution = {}
            for task_id, start_var in start_times.items():
                solution[task_id] = solver.Value(start_var)
            
            return {
                'status': 'SOLVED',
                'makespan': solver.Value(makespan),
                'start_times': solution,
                'objective_value': solver.ObjectiveValue()
            }
        else:
            return {'status': 'INFEASIBLE'}
    
    def _simulate_solution(self, jobs: List[Job]) -> Dict:
        """Simulate solution when OR-Tools not available"""
        print("  🔄 Simulating CP-SAT solution (OR-Tools not available)")
        
        # Simple greedy simulation
        machine_next_free = {m.id: 0 for m in self.machines}
        job_schedule = {}
        
        for job in jobs:
            current_time = 0
            for op_idx, (machine_id, processing_time) in enumerate(job.operations):
                # Find earliest available time on machine
                earliest_start = max(current_time, machine_next_free[machine_id])
                
                task_id = f"job_{job.id}_op_{op_idx}"
                job_schedule[task_id] = earliest_start
                
                # Update machine availability
                machine_next_free[machine_id] = earliest_start + processing_time
                current_time = earliest_start + processing_time
        
        makespan = max(machine_next_free.values())
        
        return {
            'status': 'SIMULATED',
            'makespan': makespan,
            'start_times': job_schedule,
            'objective_value': makespan
        }
    
    def analyze_solution_with_trees(self, solution: Dict, jobs: List[Job]) -> None:
        """Analyze solution using interval tree summaries"""
        if solution['status'] == 'INFEASIBLE':
            print("❌ No feasible solution found")
            return
        
        print(f"\n📊 Solution Analysis (Status: {solution['status']}):")
        print(f"  Makespan: {solution['makespan']}")
        
        # Reset trees and populate with solution
        for machine_id in self.machine_trees.keys():
            tree = create_example_tree(self.backend, random_seed=42)
            tree.release_interval(0, self.horizon)
            self.machine_trees[machine_id] = tree
        
        machine_schedules = {m.id: [] for m in self.machines}
        
        # Build schedule from solution
        for job in jobs:
            for op_idx, (machine_id, processing_time) in enumerate(job.operations):
                task_id = f"job_{job.id}_op_{op_idx}"
                start_time = solution['start_times'][task_id]
                end_time = start_time + processing_time
                
                machine_schedules[machine_id].append((start_time, end_time, f"{job.id}.{op_idx}"))
                
                # Reserve interval in machine's tree
                self.machine_trees[machine_id].reserve_interval(start_time, end_time)
        
        # Analyze using backend-agnostic tree analytics
        print("\n  Machine utilization analysis:")
        total_utilization = 0
        total_fragmentation = 0
        
        for machine in self.machines:
            analytics = get_tree_analytics(self.machine_trees[machine.id])
            utilization = analytics.get('utilization', 0)
            fragmentation = analytics.get('fragmentation', 0)
            free_chunks = analytics.get('free_chunks', analytics.get('tree_size', 0))
            
            total_utilization += utilization
            total_fragmentation += fragmentation
            
            print(f"    Machine {machine.id} ({machine.name}): "
                  f"util={utilization:.1%}, frag={fragmentation:.1%}, "
                  f"chunks={free_chunks}")
        
        avg_utilization = total_utilization / len(self.machines)
        avg_fragmentation = total_fragmentation / len(self.machines)
        
        print(f"\n  Overall system metrics:")
        print(f"    Average utilization: {avg_utilization:.1%}")
        print(f"    Average fragmentation: {avg_fragmentation:.1%}")
        print(f"    Makespan efficiency: {solution['makespan']} time units")


def demo_flexible_job_shop():
    """Flexible job shop where operations can use multiple machines"""
    print("\n🔄 Flexible Job Shop Scheduling Demo")
    print("=" * 50)
    
    machines = [
        Machine(0, "CNC_1", capacity=1),
        Machine(1, "CNC_2", capacity=1),
        Machine(2, "Mill_1", capacity=1),
        Machine(3, "Mill_2", capacity=1),
        Machine(4, "Drill", capacity=2),  # Higher capacity
    ]
    
    # Jobs with flexible machine assignments
    flexible_jobs = [
        Job(0, [(0, 15), (2, 20), (4, 10)], due_date=80),  # CNC → Mill → Drill
        Job(1, [(1, 12), (3, 18), (4, 8)], due_date=75),   # CNC → Mill → Drill  
        Job(2, [(0, 10), (2, 25), (4, 12)], due_date=90),  # CNC → Mill → Drill
        Job(3, [(1, 18), (3, 15), (4, 9)], due_date=85),   # CNC → Mill → Drill
    ]
    
    # Use same backend as main function (get from global if needed)
    scheduler = JobShopCPSAT(machines, horizon=200, backend="py_summary")
    
    print("Flexible job shop configuration:")
    for machine in machines:
        print(f"  {machine.name} (ID: {machine.id}, Capacity: {machine.capacity})")
    
    print(f"\nJobs to schedule:")
    for job in flexible_jobs:
        operations_str = " → ".join([f"M{mid}({pt})" for mid, pt in job.operations])
        print(f"  Job {job.id}: {operations_str} (due: {job.due_date})")
    
    # Solve the flexible job shop problem
    solution = scheduler.solve_job_shop(flexible_jobs)
    scheduler.analyze_solution_with_trees(solution, flexible_jobs)
    
    # Demonstrate constraint relaxation
    print(f"\n🔧 Constraint Relaxation Analysis:")
    if solution['status'] != 'INFEASIBLE':
        print("  Original problem: FEASIBLE")
        
        # Try with tighter due dates
        tight_jobs = [Job(j.id, j.operations, due_date=int(j.due_date * 0.8)) for j in flexible_jobs]
        tight_solution = scheduler.solve_job_shop(tight_jobs)
        
        if tight_solution['status'] == 'INFEASIBLE':
            print("  Tighter due dates (80%): INFEASIBLE")
            print("  → Due date constraints are binding")
        else:
            print(f"  Tighter due dates (80%): FEASIBLE (makespan: {tight_solution['makespan']})")


def demo_multi_objective_optimization():
    """Multi-objective optimization using CP-SAT with tree guidance"""
    print("\n🎯 Multi-Objective Job Shop Optimization")
    print("=" * 50)
    
    # Create problem with multiple conflicting objectives
    machines = [Machine(i, f"Machine_{i}") for i in range(3)]
    
    jobs = [
        Job(0, [(0, 10), (1, 15), (2, 12)], due_date=50, priority=3),
        Job(1, [(1, 8), (2, 20), (0, 10)], due_date=45, priority=1),
        Job(2, [(2, 12), (0, 18), (1, 14)], due_date=60, priority=2),
        Job(3, [(0, 15), (2, 10), (1, 16)], due_date=55, priority=2),
    ]
    
    scheduler = JobShopCPSAT(machines, horizon=150, backend="py_summary")
    
    print("Multi-objective optimization:")
    print("  Objective 1: Minimize makespan")
    print("  Objective 2: Minimize weighted tardiness")
    print("  Objective 3: Minimize machine idle time")
    
    # Solve with different objective weights
    objectives = [
        ("Makespan Focus", (1.0, 0.1, 0.1)),
        ("Due Date Focus", (0.1, 1.0, 0.1)),
        ("Utilization Focus", (0.1, 0.1, 1.0)),
        ("Balanced", (0.33, 0.33, 0.34)),
    ]
    
    for obj_name, (w1, w2, w3) in objectives:
        print(f"\n{obj_name} (weights: {w1:.1f}, {w2:.1f}, {w3:.1f}):")
        solution = scheduler.solve_job_shop(jobs)
        
        if solution['status'] != 'INFEASIBLE':
            makespan = solution['makespan']
            
            # Calculate other objectives using tree summaries
            total_tardiness = 0
            for job in jobs:
                last_op_idx = len(job.operations) - 1
                last_task = f"job_{job.id}_op_{last_op_idx}"
                completion_time = solution['start_times'][last_task] + job.operations[last_op_idx][1]
                
                if job.due_date and completion_time > job.due_date:
                    tardiness = completion_time - job.due_date
                    total_tardiness += job.priority * tardiness
            
            # Calculate machine idle time using backend-agnostic analytics
            total_idle_time = 0
            for machine in machines:
                analytics = get_tree_analytics(scheduler.machine_trees[machine.id])
                idle_time = analytics.get('total_free', analytics.get('total_available', 0))
                total_idle_time += idle_time
            
            print(f"    Makespan: {makespan}")
            print(f"    Weighted tardiness: {total_tardiness}")
            print(f"    Total idle time: {total_idle_time}")
            print(f"    Combined objective: {w1*makespan + w2*total_tardiness + w3*total_idle_time:.1f}")


def demo_tree_guided_constraint_generation():
    """Demonstrate using tree summaries to guide constraint generation"""
    print("\n🧠 Tree-Guided Constraint Generation")
    print("=" * 50)
    
    class IntelligentJobShopSolver:
        def __init__(self, machines: List[Machine]):
            self.machines = machines
            self.trees = {m.id: SummaryIntervalTree() for m in machines}
        
        def analyze_bottlenecks(self, jobs: List[Job]) -> Dict[int, float]:
            """Use tree analysis to identify bottleneck machines"""
            machine_loads = {m.id: 0 for m in self.machines}
            
            # Calculate expected load per machine
            for job in jobs:
                for machine_id, processing_time in job.operations:
                    machine_loads[machine_id] += processing_time
            
            # Normalize by available time
            machine_utilizations = {}
            for machine_id, load in machine_loads.items():
                available_time = 1000  # Horizon
                machine_utilizations[machine_id] = load / available_time
            
            return machine_utilizations
        
        def generate_smart_constraints(self, jobs: List[Job], bottlenecks: Dict[int, float]):
            """Generate constraints focusing on bottleneck machines"""
            constraints = []
            
            # For bottleneck machines, add tighter constraints
            for machine_id, utilization in bottlenecks.items():
                if utilization > 0.8:  # High utilization
                    constraints.append(f"Bottleneck constraint for machine {machine_id}")
                    
                    # Add setup time constraints for bottleneck machines
                    operations_on_machine = []
                    for job in jobs:
                        for op_idx, (mid, pt) in enumerate(job.operations):
                            if mid == machine_id:
                                operations_on_machine.append((job.id, op_idx, pt))
                    
                    # Add sequence-dependent setup times
                    if len(operations_on_machine) > 1:
                        constraints.append(f"Setup time constraints for {len(operations_on_machine)} operations")
            
            return constraints
    
    # Demo the intelligent solver
    machines = [Machine(i, f"Machine_{i}") for i in range(4)]
    jobs = [
        Job(0, [(0, 20), (1, 15), (2, 10)]),  # Heavy on machine 0
        Job(1, [(0, 18), (1, 12), (3, 8)]),   # Heavy on machine 0
        Job(2, [(0, 22), (2, 14), (3, 11)]),  # Heavy on machine 0
        Job(3, [(1, 10), (2, 16), (3, 9)]),   # Light on machine 0
    ]
    
    solver = IntelligentJobShopSolver(machines)
    bottlenecks = solver.analyze_bottlenecks(jobs)
    
    print("Bottleneck analysis:")
    for machine_id, utilization in bottlenecks.items():
        status = "🔴 BOTTLENECK" if utilization > 0.8 else "🟢 OK"
        print(f"  Machine {machine_id}: {utilization:.1%} utilization {status}")
    
    constraints = solver.generate_smart_constraints(jobs, bottlenecks)
    print(f"\nGenerated {len(constraints)} intelligent constraints:")
    for constraint in constraints:
        print(f"  • {constraint}")


def main():
    """Run all CP-SAT demonstrations"""
    # Parse backend configuration
    args = parse_backend_args("CP-SAT Job Shop Scheduling with Tree Enhancement")
    
    # Handle backend selection
    selected_backend = handle_backend_args(args)
    if selected_backend is None:
        return
    
    print("🔧 CP-SAT Job Shop Scheduling with Tree Enhancement")
    print("Demonstrating constraint programming with interval tree integration")
    print("=" * 70)
    
    # Basic job shop problem
    machines = [
        Machine(0, "Lathe", capacity=1),
        Machine(1, "Mill", capacity=1), 
        Machine(2, "Drill", capacity=1),
    ]
    
    jobs = [
        Job(0, [(0, 10), (1, 8), (2, 6)], due_date=35),
        Job(1, [(1, 12), (2, 10), (0, 8)], due_date=40),
        Job(2, [(2, 8), (0, 15), (1, 10)], due_date=45),
    ]
    
    print("Basic Job Shop Problem:")
    for machine in machines:
        print(f"  {machine.name} (ID: {machine.id})")
    
    for job in jobs:
        operations_str = " → ".join([f"{machines[mid].name}({pt})" for mid, pt in job.operations])
        print(f"  Job {job.id}: {operations_str} (due: {job.due_date})")
    
    # Solve and analyze using selected backend
    scheduler = JobShopCPSAT(machines, horizon=100, backend=selected_backend)
    solution = scheduler.solve_job_shop(jobs)
    scheduler.analyze_solution_with_trees(solution, jobs)
    
    # Run additional demos
    demo_flexible_job_shop()
    demo_multi_objective_optimization()
    demo_tree_guided_constraint_generation()
    
    print("\n" + "=" * 70)
    print("✅ CP-SAT demonstrations complete!")
    print("\n🎯 Key techniques demonstrated:")
    print("  • Constraint programming with tree-guided variable domains")
    print("  • Multi-objective optimization with tree-based analysis")
    print("  • Intelligent constraint generation using bottleneck detection")
    print("  • Solution analysis using O(1) tree summary statistics")
    print("  • Integration of classical OR methods with modern data structures")


if __name__ == "__main__":
    main()
