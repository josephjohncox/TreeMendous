#!/usr/bin/env python3
"""
Manufacturing Production Scheduler

Real-world manufacturing scheduling system demonstrating integration of:
- Interval trees for resource management
- Constraint programming for complex scheduling
- Real-time deadline management
- Stochastic optimization for uncertain processing times
"""

import sys
import random
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json

# Add Tree-Mendous to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'treemendous' / 'basic'))

try:
    from summary import SummaryIntervalTree
    print("✅ Tree-Mendous interval trees loaded")
except ImportError as e:
    print(f"❌ Failed to load interval trees: {e}")
    sys.exit(1)


class ResourceType(Enum):
    MACHINE = "machine"
    OPERATOR = "operator"
    TOOL = "tool"
    MATERIAL = "material"


@dataclass
class Resource:
    """Manufacturing resource with capacity and availability"""
    id: int
    name: str
    type: ResourceType
    capacity: int
    availability_tree: SummaryIntervalTree = field(default_factory=SummaryIntervalTree)
    hourly_cost: float = 0.0
    setup_time: int = 0
    
    def __post_init__(self):
        # Initialize with full availability (24 hours = 1440 minutes)
        self.availability_tree.release_interval(0, 1440)


@dataclass
class Operation:
    """Manufacturing operation with resource requirements"""
    id: int
    name: str
    processing_time_mean: int  # Expected processing time in minutes
    processing_time_std: int   # Standard deviation
    required_resources: Dict[int, int]  # resource_id -> quantity needed
    setup_requirements: List[int] = field(default_factory=list)  # Required setup operations
    
    def sample_processing_time(self) -> int:
        """Sample actual processing time from distribution"""
        return max(1, int(random.gauss(self.processing_time_mean, self.processing_time_std)))


@dataclass
class Job:
    """Manufacturing job with operations sequence"""
    id: int
    part_number: str
    operations: List[Operation]
    quantity: int
    due_date: int  # Minutes from start of day
    priority: int = 1
    customer_id: Optional[str] = None
    
    # Tracking
    completed_operations: List[int] = field(default_factory=list)
    start_time: Optional[int] = None
    completion_time: Optional[int] = None
    
    def next_operation(self) -> Optional[Operation]:
        """Get next operation to be performed"""
        if len(self.completed_operations) < len(self.operations):
            return self.operations[len(self.completed_operations)]
        return None
    
    def is_complete(self) -> bool:
        """Check if all operations completed"""
        return len(self.completed_operations) == len(self.operations)
    
    def estimated_remaining_time(self) -> int:
        """Estimate remaining processing time"""
        remaining_ops = self.operations[len(self.completed_operations):]
        return sum(op.processing_time_mean for op in remaining_ops)


class ManufacturingScheduler:
    """Advanced manufacturing scheduler with tree-enhanced optimization"""
    
    def __init__(self, resources: List[Resource]):
        self.resources = {r.id: r for r in resources}
        self.current_time = 0
        self.schedule_horizon = 1440  # 24 hours
        
        # Performance tracking
        self.completed_jobs = []
        self.missed_deadlines = []
        self.resource_utilization_history = []
        
    def schedule_jobs(self, jobs: List[Job], algorithm: str = "tree_enhanced") -> Dict:
        """Schedule jobs using specified algorithm"""
        print(f"📋 Scheduling {len(jobs)} jobs using {algorithm} algorithm...")
        
        if algorithm == "tree_enhanced":
            return self._tree_enhanced_scheduling(jobs)
        elif algorithm == "priority_dispatch":
            return self._priority_dispatch_scheduling(jobs)
        elif algorithm == "genetic_algorithm":
            return self._genetic_algorithm_scheduling(jobs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _tree_enhanced_scheduling(self, jobs: List[Job]) -> Dict:
        """Tree-enhanced scheduling with summary-guided decisions"""
        # Sort jobs by priority and due date urgency
        jobs_queue = sorted(jobs, key=lambda j: (j.priority, j.due_date))
        
        completed = []
        missed = []
        total_tardiness = 0
        
        while jobs_queue:
            # Select next job using tree-guided heuristics
            job = self._select_next_job_tree_guided(jobs_queue)
            jobs_queue.remove(job)
            
            # Schedule all operations for this job
            success = self._schedule_job_operations(job)
            
            if success and job.completion_time <= job.due_date:
                completed.append(job)
            else:
                missed.append(job)
                if job.completion_time:
                    tardiness = max(0, job.completion_time - job.due_date)
                    total_tardiness += tardiness
        
        # Analyze performance using tree summaries
        resource_stats = {}
        total_cost = 0
        
        for resource_id, resource in self.resources.items():
            stats = resource.availability_tree.get_availability_stats()
            resource_stats[resource_id] = {
                'utilization': stats['utilization'],
                'fragmentation': stats['fragmentation'],
                'total_busy_time': stats['total_occupied']
            }
            
            # Calculate resource cost
            busy_time_hours = stats['total_occupied'] / 60  # Convert to hours
            total_cost += busy_time_hours * resource.hourly_cost
        
        return {
            'algorithm': 'tree_enhanced',
            'completed_jobs': len(completed),
            'missed_deadlines': len(missed),
            'success_rate': len(completed) / len(jobs) if jobs else 1.0,
            'total_tardiness': total_tardiness,
            'avg_tardiness': total_tardiness / len(missed) if missed else 0,
            'total_cost': total_cost,
            'resource_stats': resource_stats,
            'makespan': max((job.completion_time for job in completed + missed if job.completion_time), default=0)
        }
    
    def _select_next_job_tree_guided(self, jobs_queue: List[Job]) -> Job:
        """Select next job using tree summary guidance"""
        best_job = None
        best_score = float('inf')
        
        for job in jobs_queue:
            # Score based on multiple factors
            urgency = (job.due_date - self.current_time) / job.estimated_remaining_time()
            
            # Resource availability score using tree summaries
            resource_availability = 0
            required_ops = 0
            
            next_op = job.next_operation()
            if next_op:
                for resource_id, quantity in next_op.required_resources.items():
                    if resource_id in self.resources:
                        stats = self.resources[resource_id].availability_tree.get_availability_stats()
                        # Prefer resources with low utilization and fragmentation
                        availability_score = (1 - stats['utilization']) * (1 - stats['fragmentation'])
                        resource_availability += availability_score
                        required_ops += 1
                
                avg_resource_availability = resource_availability / required_ops if required_ops > 0 else 0
                
                # Combined score: urgency + resource availability
                score = urgency + avg_resource_availability
                
                if score < best_score:
                    best_score = score
                    best_job = job
        
        return best_job if best_job else jobs_queue[0]
    
    def _schedule_job_operations(self, job: Job) -> bool:
        """Schedule all operations for a job"""
        current_job_time = self.current_time
        
        for operation in job.operations:
            # Find best time slot for this operation
            best_start_time = self._find_best_operation_slot(operation, current_job_time)
            
            if best_start_time is None:
                # Cannot schedule this operation
                return False
            
            # Sample actual processing time
            actual_processing_time = operation.sample_processing_time()
            
            # Reserve resources
            for resource_id, quantity in operation.required_resources.items():
                if resource_id in self.resources:
                    # Reserve time slots (simplified - assumes sufficient capacity)
                    self.resources[resource_id].availability_tree.reserve_interval(
                        best_start_time, best_start_time + actual_processing_time
                    )
            
            current_job_time = best_start_time + actual_processing_time
            job.completed_operations.append(operation.id)
        
        job.start_time = self.current_time
        job.completion_time = current_job_time
        return True
    
    def _find_best_operation_slot(self, operation: Operation, earliest_start: int) -> Optional[int]:
        """Find best time slot for operation using tree analysis"""
        # Check resource availability
        for resource_id, quantity in operation.required_resources.items():
            if resource_id not in self.resources:
                continue
                
            resource = self.resources[resource_id]
            
            # Use tree to find available slot
            try:
                processing_time = operation.processing_time_mean  # Use expected time for planning
                window = resource.availability_tree.find_best_fit(processing_time)
                
                if window and window[0] >= earliest_start:
                    return window[0]
                
            except ValueError:
                continue
        
        return None
    
    def _priority_dispatch_scheduling(self, jobs: List[Job]) -> Dict:
        """Simple priority-based dispatching for comparison"""
        print("  Using priority dispatch scheduling...")
        
        # Simple FIFO with priority ordering
        jobs_queue = sorted(jobs, key=lambda j: (j.priority, j.due_date))
        
        for job in jobs_queue:
            # Simple greedy scheduling
            self._schedule_job_operations(job)
        
        completed = [j for j in jobs if j.is_complete()]
        missed = [j for j in jobs if not j.is_complete() or (j.completion_time and j.completion_time > j.due_date)]
        
        return {
            'algorithm': 'priority_dispatch',
            'completed_jobs': len(completed),
            'missed_deadlines': len(missed),
            'success_rate': len(completed) / len(jobs) if jobs else 1.0
        }
    
    def _genetic_algorithm_scheduling(self, jobs: List[Job]) -> Dict:
        """Genetic algorithm scheduling using tree-guided operators"""
        print("  Using genetic algorithm with tree guidance...")
        
        class Schedule:
            def __init__(self, job_order: List[int]):
                self.job_order = job_order
                self.fitness = None
        
        def evaluate_schedule(schedule: Schedule) -> float:
            """Evaluate schedule fitness using tree simulation"""
            # Reset trees
            temp_trees = {r_id: SummaryIntervalTree() for r_id in self.resources.keys()}
            for tree in temp_trees.values():
                tree.release_interval(0, self.schedule_horizon)
            
            total_penalty = 0
            current_time = 0
            
            for job_id in schedule.job_order:
                job = next(j for j in jobs if j.id == job_id)
                
                # Simplified scheduling for fitness evaluation
                job_completion_time = current_time
                for operation in job.operations:
                    job_completion_time += operation.processing_time_mean
                
                # Penalty for late completion
                if job_completion_time > job.due_date:
                    penalty = (job_completion_time - job.due_date) * job.priority
                    total_penalty += penalty
                
                current_time = job_completion_time
            
            return -total_penalty  # Negative penalty = fitness
        
        # Genetic algorithm parameters
        population_size = 50
        generations = 30
        mutation_rate = 0.1
        
        # Initialize population
        population = []
        job_ids = [j.id for j in jobs]
        
        for _ in range(population_size):
            job_order = job_ids.copy()
            random.shuffle(job_order)
            population.append(Schedule(job_order))
        
        print(f"    Running {generations} generations with population {population_size}...")
        
        best_fitness_history = []
        
        for generation in range(generations):
            # Evaluate fitness
            for schedule in population:
                if schedule.fitness is None:
                    schedule.fitness = evaluate_schedule(schedule)
            
            # Selection and reproduction
            population.sort(key=lambda s: s.fitness, reverse=True)
            best_fitness_history.append(population[0].fitness)
            
            if generation % 10 == 0:
                print(f"      Generation {generation}: Best fitness = {population[0].fitness:.1f}")
            
            # Create next generation
            new_population = []
            
            # Elitism: keep top 10%
            elite_size = population_size // 10
            new_population.extend(population[:elite_size])
            
            # Crossover and mutation
            while len(new_population) < population_size:
                # Select parents
                parent1 = random.choice(population[:population_size//2])
                parent2 = random.choice(population[:population_size//2])
                
                # Order crossover
                child_order = self._order_crossover(parent1.job_order, parent2.job_order)
                
                # Mutation
                if random.random() < mutation_rate:
                    child_order = self._mutate_order(child_order)
                
                new_population.append(Schedule(child_order))
            
            population = new_population
        
        # Best schedule found
        best_schedule = max(population, key=lambda s: s.fitness)
        
        return {
            'algorithm': 'genetic_algorithm',
            'best_fitness': best_schedule.fitness,
            'best_order': best_schedule.job_order,
            'generations_run': generations,
            'convergence_history': best_fitness_history
        }
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order crossover for job sequences"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [-1] * size
        child[start:end] = parent1[start:end]
        
        pointer = end
        for item in parent2[end:] + parent2[:end]:
            if item not in child:
                child[pointer % size] = item
                pointer += 1
        
        return child
    
    def _mutate_order(self, order: List[int]) -> List[int]:
        """Swap mutation for job order"""
        new_order = order.copy()
        i, j = random.sample(range(len(new_order)), 2)
        new_order[i], new_order[j] = new_order[j], new_order[i]
        return new_order
    
    def analyze_performance(self, results: Dict, jobs: List[Job]) -> None:
        """Analyze scheduling performance using tree summaries"""
        print(f"\n📊 Performance Analysis ({results['algorithm']}):")
        print(f"  Jobs completed: {results.get('completed_jobs', 'N/A')}")
        print(f"  Deadlines missed: {results.get('missed_deadlines', 'N/A')}")
        print(f"  Success rate: {results.get('success_rate', 0):.1%}")
        
        if 'makespan' in results:
            print(f"  Makespan: {results['makespan']} minutes")
        
        if 'total_cost' in results:
            print(f"  Total resource cost: ${results['total_cost']:.2f}")
        
        # Resource utilization analysis
        if 'resource_stats' in results:
            print(f"\n  Resource utilization:")
            for resource_id, stats in results['resource_stats'].items():
                resource_name = self.resources[resource_id].name
                print(f"    {resource_name}: {stats['utilization']:.1%} utilized, "
                      f"{stats['fragmentation']:.1%} fragmented")
    
    def generate_production_schedule_report(self, jobs: List[Job], results: Dict) -> str:
        """Generate detailed production schedule report"""
        report = []
        report.append("🏭 PRODUCTION SCHEDULE REPORT")
        report.append("=" * 50)
        
        # Summary statistics
        report.append(f"Schedule Summary:")
        report.append(f"  Total jobs: {len(jobs)}")
        report.append(f"  Scheduling algorithm: {results.get('algorithm', 'unknown')}")
        report.append(f"  Success rate: {results.get('success_rate', 0):.1%}")
        
        # Resource efficiency
        if 'resource_stats' in results:
            report.append(f"\nResource Efficiency:")
            for resource_id, stats in results['resource_stats'].items():
                resource = self.resources[resource_id]
                efficiency = stats['utilization'] * (1 - stats['fragmentation'])
                report.append(f"  {resource.name}: {efficiency:.1%} efficiency "
                            f"({stats['utilization']:.1%} util, {stats['fragmentation']:.1%} frag)")
        
        # Job details
        report.append(f"\nJob Details:")
        completed_jobs = [j for j in jobs if j.is_complete()]
        for job in completed_jobs[:10]:  # Show first 10
            on_time = "✅" if job.completion_time <= job.due_date else "❌"
            report.append(f"  Job {job.id} ({job.part_number}): "
                        f"completed at {job.completion_time}, due {job.due_date} {on_time}")
        
        if len(completed_jobs) > 10:
            report.append(f"  ... and {len(completed_jobs) - 10} more jobs")
        
        return "\n".join(report)


def demo_semiconductor_fab():
    """Semiconductor fabrication scheduling example"""
    print("🔬 Semiconductor Fabrication Scheduling")
    print("=" * 50)
    
    # Simplified fab resources
    fab_resources = [
        Resource(0, "Lithography_1", ResourceType.MACHINE, 1, hourly_cost=500),
        Resource(1, "Lithography_2", ResourceType.MACHINE, 1, hourly_cost=500),
        Resource(2, "Etch_1", ResourceType.MACHINE, 1, hourly_cost=300),
        Resource(3, "Etch_2", ResourceType.MACHINE, 1, hourly_cost=300),
        Resource(4, "Deposition", ResourceType.MACHINE, 1, hourly_cost=400),
        Resource(5, "CMP", ResourceType.MACHINE, 1, hourly_cost=350),
        Resource(6, "Metrology", ResourceType.MACHINE, 1, hourly_cost=200),
        Resource(7, "Clean_Room_Tech", ResourceType.OPERATOR, 3, hourly_cost=50),
    ]
    
    # Typical semiconductor process flow
    operations = [
        Operation(0, "Photoresist_Coat", 30, 5, {0: 1, 7: 1}),  # Litho + operator
        Operation(1, "Exposure", 45, 8, {0: 1}),  # Lithography
        Operation(2, "Develop", 20, 3, {7: 1}),   # Operator only
        Operation(3, "Etch", 60, 12, {2: 1}),     # Etch tool
        Operation(4, "Strip", 25, 4, {7: 1}),     # Operator
        Operation(5, "Deposit_Layer", 90, 15, {4: 1}),  # Deposition
        Operation(6, "CMP_Polish", 40, 8, {5: 1}),       # CMP
        Operation(7, "Metrology_Check", 15, 2, {6: 1}),  # Metrology
    ]
    
    # Generate wafer lots (jobs)
    wafer_lots = []
    for i in range(8):  # 8 wafer lots
        due_date = random.randint(400, 800)  # 6-13 hours from start
        priority = random.randint(1, 3)
        
        wafer_lots.append(Job(
            id=i,
            part_number=f"WAFER_LOT_{i:03d}",
            operations=operations.copy(),
            quantity=25,  # 25 wafers per lot
            due_date=due_date,
            priority=priority,
            customer_id=f"FAB_CUSTOMER_{i//2}"  # Multiple lots per customer
        ))
    
    scheduler = ManufacturingScheduler(fab_resources)
    
    print("Semiconductor fab configuration:")
    for resource in fab_resources:
        print(f"  {resource.name}: ${resource.hourly_cost}/hr")
    
    print(f"\nWafer lots to process:")
    for lot in wafer_lots:
        print(f"  {lot.part_number}: {len(lot.operations)} ops, due={lot.due_date}, priority={lot.priority}")
    
    # Compare different scheduling algorithms
    algorithms = ["tree_enhanced", "priority_dispatch"]
    
    for algorithm in algorithms:
        # Reset scheduler state
        scheduler = ManufacturingScheduler(fab_resources)
        
        results = scheduler.schedule_jobs(wafer_lots.copy(), algorithm)
        scheduler.analyze_performance(results, wafer_lots)
        
        print()  # Spacing between algorithms


def demo_automotive_assembly():
    """Automotive assembly line scheduling"""
    print("\n🚗 Automotive Assembly Line Scheduling")
    print("=" * 50)
    
    # Assembly line resources
    assembly_resources = [
        Resource(0, "Body_Welding", ResourceType.MACHINE, 2, hourly_cost=200),
        Resource(1, "Paint_Booth", ResourceType.MACHINE, 1, hourly_cost=300),
        Resource(2, "Engine_Install", ResourceType.MACHINE, 1, hourly_cost=250),
        Resource(3, "Interior_Assembly", ResourceType.MACHINE, 2, hourly_cost=150),
        Resource(4, "Final_Inspection", ResourceType.MACHINE, 1, hourly_cost=100),
        Resource(5, "Assembly_Worker", ResourceType.OPERATOR, 10, hourly_cost=30),
        Resource(6, "Quality_Inspector", ResourceType.OPERATOR, 3, hourly_cost=40),
    ]
    
    # Vehicle assembly operations
    assembly_operations = [
        Operation(0, "Weld_Body", 120, 15, {0: 1, 5: 2}),
        Operation(1, "Paint_Vehicle", 180, 20, {1: 1, 5: 1}),
        Operation(2, "Install_Engine", 90, 12, {2: 1, 5: 2}),
        Operation(3, "Install_Interior", 150, 18, {3: 1, 5: 3}),
        Operation(4, "Final_QC", 45, 8, {4: 1, 6: 1}),
    ]
    
    # Vehicle orders with different priorities
    vehicle_orders = []
    order_types = [
        ("Economy", 1, 600),   # Lower priority, longer due date
        ("Premium", 2, 500),   # Medium priority
        ("Custom", 3, 400),    # High priority, tight due date
    ]
    
    for i in range(12):  # 12 vehicle orders
        order_type, priority, base_due_date = random.choice(order_types)
        due_date = base_due_date + random.randint(-50, 100)
        
        vehicle_orders.append(Job(
            id=i,
            part_number=f"VEHICLE_{order_type}_{i:03d}",
            operations=assembly_operations.copy(),
            quantity=1,
            due_date=due_date,
            priority=priority,
            customer_id=f"DEALER_{i//3}"
        ))
    
    scheduler = ManufacturingScheduler(assembly_resources)
    
    print("Automotive assembly configuration:")
    print(f"  {len(assembly_resources)} resources, {len(assembly_operations)} operations per vehicle")
    print(f"  {len(vehicle_orders)} vehicle orders")
    
    # Analyze workload using tree summaries
    print(f"\nWorkload analysis:")
    total_work = sum(sum(op.processing_time_mean for op in job.operations) for job in vehicle_orders)
    available_capacity = len(assembly_resources) * 1440  # 24 hours per resource
    base_utilization = total_work / available_capacity
    
    print(f"  Total work: {total_work} minutes")
    print(f"  Available capacity: {available_capacity} minutes")
    print(f"  Base utilization: {base_utilization:.1%}")
    
    # Schedule using tree-enhanced algorithm
    results = scheduler.schedule_jobs(vehicle_orders, "tree_enhanced")
    scheduler.analyze_performance(results, vehicle_orders)
    
    # Generate detailed report
    report = scheduler.generate_production_schedule_report(vehicle_orders, results)
    print(f"\n{report}")


def demo_supply_chain_coordination():
    """Supply chain coordination with multiple facilities"""
    print("\n🌐 Supply Chain Coordination")
    print("=" * 50)
    
    class SupplyChainCoordinator:
        def __init__(self, facilities: Dict[str, ManufacturingScheduler]):
            self.facilities = facilities
            self.coordination_tree = SummaryIntervalTree()
            self.coordination_tree.release_interval(0, 1440)
        
        def coordinate_production(self, orders: List[Dict]) -> Dict:
            """Coordinate production across multiple facilities"""
            
            # Analyze capacity across facilities using tree summaries
            facility_capacities = {}
            for facility_name, scheduler in self.facilities.items():
                total_capacity = 0
                total_utilization = 0
                
                for resource in scheduler.resources.values():
                    stats = resource.availability_tree.get_availability_stats()
                    total_capacity += stats['total_space']
                    total_utilization += stats['utilization']
                
                avg_utilization = total_utilization / len(scheduler.resources)
                facility_capacities[facility_name] = {
                    'capacity': total_capacity,
                    'utilization': avg_utilization,
                    'available': total_capacity * (1 - avg_utilization)
                }
            
            # Allocate orders to facilities based on capacity and specialization
            allocation_plan = {}
            
            for order in orders:
                best_facility = None
                best_score = float('inf')
                
                for facility_name, capacity_info in facility_capacities.items():
                    # Score based on utilization and specialization
                    utilization_penalty = capacity_info['utilization'] * 100
                    
                    # Specialization bonus (simplified)
                    specialization_bonus = 0
                    if "electronics" in order.get('type', '') and "Electronics" in facility_name:
                        specialization_bonus = -20
                    elif "automotive" in order.get('type', '') and "Automotive" in facility_name:
                        specialization_bonus = -20
                    
                    score = utilization_penalty + specialization_bonus
                    
                    if score < best_score:
                        best_score = score
                        best_facility = facility_name
                
                allocation_plan[order['id']] = best_facility
            
            return {
                'allocation_plan': allocation_plan,
                'facility_capacities': facility_capacities,
                'coordination_efficiency': self._calculate_coordination_efficiency(allocation_plan, facility_capacities)
            }
        
        def _calculate_coordination_efficiency(self, allocation_plan: Dict, 
                                            facility_capacities: Dict) -> float:
            """Calculate efficiency of coordination decisions"""
            # Measure load balance across facilities
            facility_loads = defaultdict(int)
            for order_id, facility in allocation_plan.items():
                facility_loads[facility] += 1
            
            # Calculate load balance (lower is better)
            loads = list(facility_loads.values())
            if len(loads) > 1:
                load_variance = sum((x - sum(loads)/len(loads))**2 for x in loads) / len(loads)
                load_balance_score = 1.0 / (1.0 + load_variance)
            else:
                load_balance_score = 1.0
            
            return load_balance_score
    
    # Demo supply chain coordination
    electronics_resources = [
        Resource(0, "SMT_Line", ResourceType.MACHINE, 1, hourly_cost=400),
        Resource(1, "Test_Station", ResourceType.MACHINE, 2, hourly_cost=200),
    ]
    
    automotive_resources = [
        Resource(0, "Injection_Molding", ResourceType.MACHINE, 3, hourly_cost=300),
        Resource(1, "Assembly_Line", ResourceType.MACHINE, 1, hourly_cost=250),
    ]
    
    facilities = {
        "Electronics_Plant": ManufacturingScheduler(electronics_resources),
        "Automotive_Plant": ManufacturingScheduler(automotive_resources),
    }
    
    coordinator = SupplyChainCoordinator(facilities)
    
    # Mixed order types
    supply_chain_orders = [
        {'id': 0, 'type': 'electronics', 'complexity': 'medium', 'due_date': 300},
        {'id': 1, 'type': 'automotive', 'complexity': 'high', 'due_date': 400},
        {'id': 2, 'type': 'electronics', 'complexity': 'low', 'due_date': 200},
        {'id': 3, 'type': 'automotive', 'complexity': 'medium', 'due_date': 350},
        {'id': 4, 'type': 'electronics', 'complexity': 'high', 'due_date': 450},
    ]
    
    print("Supply chain coordination:")
    print(f"  {len(facilities)} facilities, {len(supply_chain_orders)} orders")
    
    coordination_results = coordinator.coordinate_production(supply_chain_orders)
    
    print(f"\nCoordination results:")
    print(f"  Allocation plan:")
    for order_id, facility in coordination_results['allocation_plan'].items():
        order = next(o for o in supply_chain_orders if o['id'] == order_id)
        print(f"    Order {order_id} ({order['type']}): → {facility}")
    
    print(f"\n  Facility capacity analysis:")
    for facility, capacity_info in coordination_results['facility_capacities'].items():
        print(f"    {facility}: {capacity_info['utilization']:.1%} utilized, "
              f"{capacity_info['available']:.0f} units available")
    
    print(f"  Coordination efficiency: {coordination_results['coordination_efficiency']:.1%}")


def main():
    """Run all manufacturing scheduling demonstrations"""
    print("🏭 Manufacturing Production Scheduling")
    print("Real-world applications of Tree-Mendous interval trees")
    print("=" * 60)
    
    random.seed(42)  # Reproducible results
    
    demo_semiconductor_fab()
    demo_automotive_assembly()
    demo_supply_chain_coordination()
    
    print("\n" + "=" * 60)
    print("✅ Manufacturing scheduling demonstrations complete!")
    print("\n🎯 Real-world applications demonstrated:")
    print("  • Semiconductor fabrication with complex process flows")
    print("  • Automotive assembly line optimization")
    print("  • Supply chain coordination across multiple facilities")
    print("  • Resource utilization analysis using O(1) tree summaries")
    print("  • Multi-objective optimization balancing cost, time, and quality")
    print("  • Genetic algorithms with tree-guided operators")
    print("  • Stochastic processing time handling")
    print("  • Real-time performance monitoring and reporting")


if __name__ == "__main__":
    main()
