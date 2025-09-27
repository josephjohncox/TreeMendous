#!/usr/bin/env python3
"""
Bellman Iteration for Queue Network Optimization

Demonstrates dynamic programming approaches for optimizing queue networks
with stochastic processing times, due date constraints, and multi-stage
decision making using Tree-Mendous interval trees for state compression.
"""

import sys
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import math

# Optional numpy import
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠️  NumPy not available - using simplified distributions")

# Add Tree-Mendous to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'treemendous' / 'basic'))

try:
    from summary import SummaryIntervalTree
    print("✅ Tree-Mendous interval trees loaded")
except ImportError as e:
    print(f"❌ Failed to load interval trees: {e}")
    sys.exit(1)


@dataclass
class Customer:
    """Customer with due date and value"""
    id: int
    arrival_time: int
    due_date: int
    value: float
    current_machine: int = 0
    route: List[int] = field(default_factory=list)
    service_history: List[Tuple[int, int, int]] = field(default_factory=list)  # (machine, start, end)


@dataclass
class Machine:
    """Machine in production network"""
    id: int
    name: str
    service_rate: float  # customers per time unit
    service_variance: float
    queue_capacity: int = float('inf')
    
    def sample_service_time(self) -> int:
        """Sample service time from distribution"""
        # Gamma distribution with specified mean and variance
        mean = 1.0 / self.service_rate
        if self.service_variance > 0:
            if NUMPY_AVAILABLE:
                shape = mean**2 / self.service_variance
                scale = self.service_variance / mean
                return max(1, int(np.random.gamma(shape, scale)))
            else:
                # Fallback: use normal distribution
                std = math.sqrt(self.service_variance)
                return max(1, int(random.gauss(mean, std)))
        else:
            return int(mean)


class QueueNetworkDP:
    """Dynamic programming optimizer for queue networks"""
    
    def __init__(self, machines: List[Machine], routing_dag: Dict[int, List[int]]):
        self.machines = {m.id: m for m in machines}
        self.routing_dag = routing_dag  # machine_id -> list of successor machines
        self.machine_trees = {m.id: SummaryIntervalTree() for m in machines}
        
        # Initialize available time
        horizon = 1000
        for tree in self.machine_trees.values():
            tree.release_interval(0, horizon)
        
        # Value function approximation
        self.value_function = {}
        self.policy = {}
        
    def get_state_signature(self, machine_id: int, queue_length: int, 
                           customer_urgency: float) -> str:
        """Create compact state signature using tree summaries"""
        tree_stats = self.machine_trees[machine_id].get_availability_stats()
        
        # Discretize continuous values for state representation
        util_bucket = int(tree_stats['utilization'] * 10)  # 0-10
        frag_bucket = int(tree_stats['fragmentation'] * 10)  # 0-10
        queue_bucket = min(queue_length, 10)  # Cap at 10
        urgency_bucket = int(customer_urgency * 5)  # 0-5
        
        return f"{machine_id}_{queue_bucket}_{util_bucket}_{frag_bucket}_{urgency_bucket}"
    
    def bellman_iteration(self, customers: List[Customer], max_iterations: int = 100) -> Dict:
        """Bellman iteration for optimal queue control"""
        print(f"🔄 Running Bellman iteration ({max_iterations} iterations)...")
        
        # Initialize value function
        states = set()
        for machine_id in self.machines.keys():
            for queue_len in range(11):  # 0-10
                for util in range(11):  # 0-10 (scaled)
                    for frag in range(11):  # 0-10 (scaled)
                        for urgency in range(6):  # 0-5 (scaled)
                            state = f"{machine_id}_{queue_len}_{util}_{frag}_{urgency}"
                            states.add(state)
                            self.value_function[state] = 0.0
        
        print(f"  State space size: {len(states):,} states")
        
        # Bellman iteration
        for iteration in range(max_iterations):
            old_values = self.value_function.copy()
            max_change = 0.0
            
            for state in states:
                # Parse state
                parts = state.split('_')
                machine_id = int(parts[0])
                queue_length = int(parts[1])
                util_bucket = int(parts[2])
                frag_bucket = int(parts[3])
                urgency_bucket = int(parts[4])
                
                # Available actions
                actions = self._get_available_actions(machine_id, queue_length)
                
                if not actions:
                    continue
                
                # Compute Q-values for each action
                q_values = []
                for action in actions:
                    q_value = self._compute_q_value(state, action, old_values)
                    q_values.append((action, q_value))
                
                # Update value function (Bellman optimality)
                if q_values:
                    best_action, best_value = min(q_values, key=lambda x: x[1])  # Minimize cost
                    self.value_function[state] = best_value
                    self.policy[state] = best_action
                    
                    change = abs(best_value - old_values[state])
                    max_change = max(max_change, change)
            
            if iteration % 20 == 0:
                print(f"    Iteration {iteration}: max change = {max_change:.6f}")
            
            # Check convergence
            if max_change < 1e-6:
                print(f"    Converged after {iteration + 1} iterations")
                break
        
        return {
            'converged': max_change < 1e-6,
            'final_max_change': max_change,
            'iterations': iteration + 1,
            'policy_size': len(self.policy)
        }
    
    def _get_available_actions(self, machine_id: int, queue_length: int) -> List[str]:
        """Get available actions for given state"""
        actions = []
        
        if queue_length > 0:
            actions.append("serve_customer")
        
        actions.append("adjust_rate_up")
        actions.append("adjust_rate_down")
        
        # Routing actions if not sink machine
        if machine_id in self.routing_dag:
            for successor in self.routing_dag[machine_id]:
                actions.append(f"route_to_{successor}")
        
        return actions
    
    def _compute_q_value(self, state: str, action: str, value_function: Dict[str, float]) -> float:
        """Compute Q-value for state-action pair"""
        parts = state.split('_')
        machine_id = int(parts[0])
        queue_length = int(parts[1])
        util_bucket = int(parts[2])
        frag_bucket = int(parts[3])
        urgency_bucket = int(parts[4])
        
        # Immediate cost
        immediate_cost = self._compute_immediate_cost(machine_id, queue_length, action)
        
        # Expected future cost
        future_cost = 0.0
        gamma = 0.95  # Discount factor
        
        # Simulate action effects (simplified)
        if action == "serve_customer" and queue_length > 0:
            # Customer served, queue length decreases
            new_queue = max(0, queue_length - 1)
            new_util = max(0, util_bucket - 1)  # Utilization might decrease
            next_state = f"{machine_id}_{new_queue}_{new_util}_{frag_bucket}_{urgency_bucket}"
            future_cost = gamma * value_function.get(next_state, 0.0)
            
        elif "route_to" in action:
            # Routing decision - customer moves to next machine
            successor_id = int(action.split('_')[-1])
            # Estimate next state on successor machine
            next_state = f"{successor_id}_1_{util_bucket}_{frag_bucket}_{urgency_bucket}"
            future_cost = gamma * value_function.get(next_state, 0.0)
        
        return immediate_cost + future_cost
    
    def _compute_immediate_cost(self, machine_id: int, queue_length: int, action: str) -> float:
        """Compute immediate cost for taking action in state"""
        # Holding cost for customers in queue
        holding_cost = queue_length * 1.0
        
        # Action-specific costs
        if action == "serve_customer":
            return holding_cost + 0.1  # Small service cost
        elif action == "adjust_rate_up":
            return holding_cost + 2.0  # Cost of increasing rate
        elif action == "adjust_rate_down":
            return holding_cost + 0.5  # Small cost for rate decrease
        elif "route_to" in action:
            return holding_cost + 0.2  # Routing cost
        
        return holding_cost
    
    def simulate_optimal_policy(self, customers: List[Customer], simulation_time: int) -> Dict:
        """Simulate network using learned optimal policy"""
        print(f"\n🎮 Simulating optimal policy for {simulation_time} time units...")
        
        # Reset trees
        for tree in self.machine_trees.values():
            tree = SummaryIntervalTree()
            tree.release_interval(0, simulation_time)
        
        # Simulation state
        machine_queues = {m_id: [] for m_id in self.machines.keys()}
        current_time = 0
        customers_completed = []
        customers_missed = []
        
        # Sort customers by arrival time
        customers.sort(key=lambda c: c.arrival_time)
        customer_idx = 0
        
        while current_time < simulation_time and customer_idx < len(customers):
            # Add arriving customers
            while (customer_idx < len(customers) and 
                   customers[customer_idx].arrival_time <= current_time):
                customer = customers[customer_idx]
                # Start at first machine in route
                if customer.route:
                    machine_queues[customer.route[0]].append(customer)
                customer_idx += 1
            
            # Process each machine
            for machine_id, queue in machine_queues.items():
                if not queue:
                    continue
                
                # Get state and optimal action
                customer = queue[0]  # FIFO assumption
                urgency = max(0, 1.0 - (customer.due_date - current_time) / customer.due_date)
                state = self.get_state_signature(machine_id, len(queue), urgency)
                
                action = self.policy.get(state, "serve_customer")
                
                if action == "serve_customer":
                    # Serve customer
                    service_time = self.machines[machine_id].sample_service_time()
                    
                    # Reserve interval in tree
                    self.machine_trees[machine_id].reserve_interval(
                        current_time, current_time + service_time
                    )
                    
                    # Update customer
                    customer.service_history.append((machine_id, current_time, current_time + service_time))
                    
                    # Move to next machine or complete
                    current_route_idx = customer.route.index(machine_id)
                    if current_route_idx + 1 < len(customer.route):
                        # Move to next machine
                        next_machine = customer.route[current_route_idx + 1]
                        machine_queues[next_machine].append(customer)
                    else:
                        # Job completed
                        completion_time = current_time + service_time
                        if completion_time <= customer.due_date:
                            customers_completed.append((customer, completion_time))
                        else:
                            customers_missed.append((customer, completion_time))
                    
                    # Remove from current queue
                    machine_queues[machine_id].remove(customer)
            
            current_time += 1  # Advance time
        
        # Analyze results using tree summaries
        machine_stats = {}
        for machine_id, tree in self.machine_trees.items():
            stats = tree.get_availability_stats()
            machine_stats[machine_id] = stats
        
        return {
            'customers_completed': len(customers_completed),
            'customers_missed': len(customers_missed),
            'success_rate': len(customers_completed) / (len(customers_completed) + len(customers_missed))
                          if (customers_completed or customers_missed) else 1.0,
            'avg_utilization': sum(stats['utilization'] for stats in machine_stats.values()) / len(machine_stats),
            'avg_fragmentation': sum(stats['fragmentation'] for stats in machine_stats.values()) / len(machine_stats),
            'machine_stats': machine_stats
        }


def demo_simple_queue_network():
    """Simple queue network with Bellman optimization"""
    print("🏭 Simple Production Line Optimization")
    print("=" * 50)
    
    # 3-machine production line: Prep → Process → Pack
    machines = [
        Machine(0, "Preparation", service_rate=0.8, service_variance=0.1),
        Machine(1, "Processing", service_rate=0.6, service_variance=0.2),  # Bottleneck
        Machine(2, "Packaging", service_rate=1.0, service_variance=0.05),
    ]
    
    # Linear routing: 0 → 1 → 2
    routing_dag = {0: [1], 1: [2], 2: []}
    
    # Generate customers with due dates
    customers = []
    for i in range(20):
        arrival_time = random.randint(0, 50)
        due_date = arrival_time + random.randint(30, 80)  # Reasonable due dates
        value = random.uniform(10, 100)
        route = [0, 1, 2]  # All customers follow same route
        
        customers.append(Customer(i, arrival_time, due_date, value, route=route))
    
    print("Production line setup:")
    for machine in machines:
        print(f"  {machine.name}: rate={machine.service_rate:.1f}, variance={machine.service_variance:.2f}")
    
    print(f"\nCustomers: {len(customers)} with due dates ranging {min(c.due_date for c in customers)}-{max(c.due_date for c in customers)}")
    
    # Create and optimize network
    optimizer = QueueNetworkDP(machines, routing_dag)
    
    # Run Bellman iteration
    bellman_results = optimizer.bellman_iteration(customers, max_iterations=50)
    print(f"\nBellman iteration results:")
    print(f"  Converged: {'✅ YES' if bellman_results['converged'] else '❌ NO'}")
    print(f"  Iterations: {bellman_results['iterations']}")
    print(f"  Policy size: {bellman_results['policy_size']:,} state-action pairs")
    
    # Simulate using optimal policy
    simulation_results = optimizer.simulate_optimal_policy(customers, 200)
    
    print(f"\nSimulation with optimal policy:")
    print(f"  Customers completed: {simulation_results['customers_completed']}")
    print(f"  Customers missed deadlines: {simulation_results['customers_missed']}")
    print(f"  Success rate: {simulation_results['success_rate']:.1%}")
    print(f"  Average utilization: {simulation_results['avg_utilization']:.1%}")
    print(f"  Average fragmentation: {simulation_results['avg_fragmentation']:.1%}")


def demo_multi_stage_decision_making():
    """Multi-stage decision making with uncertain outcomes"""
    print("\n🎲 Multi-Stage Stochastic Optimization")
    print("=" * 50)
    
    class MultiStageOptimizer:
        def __init__(self, stages: int, decisions_per_stage: int):
            self.stages = stages
            self.decisions_per_stage = decisions_per_stage
            self.trees = [SummaryIntervalTree() for _ in range(stages)]
            
            # Initialize each stage with available capacity
            for tree in self.trees:
                tree.release_interval(0, 100)  # 100 units capacity per stage
        
        def solve_multistage_problem(self, scenarios: List[Dict]) -> Dict:
            """Solve multi-stage stochastic problem using Bellman recursion"""
            
            # State: (stage, available_capacity, demand_realization)
            # Decision: allocation amounts per stage
            
            # Work backwards from final stage
            value_functions = [{} for _ in range(self.stages)]
            policies = [{} for _ in range(self.stages)]
            
            # Terminal stage (no future decisions)
            stage = self.stages - 1
            for capacity in range(101):  # 0-100 capacity
                for demand in range(51):   # 0-50 demand
                    state = (stage, capacity, demand)
                    
                    # Terminal reward: serve as much demand as possible
                    served = min(capacity, demand)
                    revenue = served * 10  # $10 per unit served
                    cost = capacity * 1    # $1 per unit capacity
                    
                    value_functions[stage][state] = revenue - cost
                    policies[stage][state] = served
            
            # Work backwards through stages
            for stage in range(self.stages - 2, -1, -1):
                print(f"    Solving stage {stage}...")
                
                for capacity in range(101):
                    for demand in range(51):
                        state = (stage, capacity, demand)
                        
                        best_value = float('-inf')
                        best_action = 0
                        
                        # Try different allocation amounts
                        for allocation in range(min(capacity, demand) + 1):
                            # Immediate reward
                            immediate_reward = allocation * 10 - capacity * 1
                            
                            # Expected future value
                            remaining_capacity = capacity - allocation
                            future_value = 0.0
                            
                            # Average over possible future demand scenarios
                            for future_demand in range(51):
                                prob = 1.0 / 51  # Uniform distribution (simplified)
                                future_state = (stage + 1, remaining_capacity, future_demand)
                                future_value += prob * value_functions[stage + 1].get(future_state, 0)
                            
                            total_value = immediate_reward + 0.9 * future_value  # γ = 0.9
                            
                            if total_value > best_value:
                                best_value = total_value
                                best_action = allocation
                        
                        value_functions[stage][state] = best_value
                        policies[stage][state] = best_action
            
            return {
                'value_functions': value_functions,
                'policies': policies,
                'optimal_value': value_functions[0].get((0, 100, 25), 0)  # Starting state
            }
    
    # Demo multi-stage optimization
    optimizer = MultiStageOptimizer(stages=4, decisions_per_stage=10)
    
    print("Multi-stage capacity allocation problem:")
    print("  • 4 stages of production")
    print("  • Uncertain demand at each stage")
    print("  • Capacity allocation decisions")
    print("  • Objective: maximize profit over planning horizon")
    
    scenarios = [{"demand": random.randint(10, 40)} for _ in range(100)]
    results = optimizer.solve_multistage_problem(scenarios)
    
    print(f"\nOptimization results:")
    print(f"  Optimal value: ${results['optimal_value']:.2f}")
    print(f"  Policy computed for {len(results['policies'][0]):,} states per stage")
    
    # Demonstrate policy usage
    print(f"\nExample policy decisions:")
    for stage in range(min(3, len(results['policies']))):
        example_state = (stage, 80, 30)  # 80 capacity, 30 demand
        action = results['policies'][stage].get(example_state, 0)
        print(f"  Stage {stage}: With capacity=80, demand=30 → allocate {action} units")


def demo_due_date_optimization():
    """Due date optimization using Bellman iteration"""
    print("\n📅 Due Date Optimization with Stochastic Completion Times")
    print("=" * 50)
    
    class DueDateOptimizer:
        def __init__(self, machines: List[Machine]):
            self.machines = machines
            self.completion_time_distributions = {}  # Cache for efficiency
        
        def estimate_completion_time_distribution(self, customer: Customer, 
                                                current_time: int) -> Tuple[float, float]:
            """Estimate completion time mean and variance"""
            expected_time = current_time
            total_variance = 0
            
            for machine_id in customer.route[customer.current_machine:]:
                machine = self.machines[machine_id]
                
                # Expected service time
                expected_service = 1.0 / machine.service_rate
                expected_time += expected_service
                
                # Add variance
                total_variance += machine.service_variance
                
                # Add expected queueing delay (simplified M/M/1 formula)
                rho = 0.7  # Assumed utilization
                if rho < 1:
                    expected_queue_wait = rho / (machine.service_rate * (1 - rho))
                    expected_time += expected_queue_wait
                    
                    # Variance of M/M/1 waiting time
                    queue_variance = (2 - rho) / (machine.service_rate**2 * (1 - rho)**3)
                    total_variance += queue_variance
            
            return expected_time, total_variance
        
        def optimize_due_date_penalties(self, customers: List[Customer]) -> Dict:
            """Optimize to minimize due date penalties using Bellman approach"""
            
            # State: (customer_id, current_machine, time)
            # Decision: routing and priority adjustments
            
            total_penalty = 0
            completion_predictions = {}
            
            for customer in customers:
                # Estimate completion time distribution
                mean_completion, variance = self.estimate_completion_time_distribution(
                    customer, customer.arrival_time
                )
                
                completion_predictions[customer.id] = {
                    'mean': mean_completion,
                    'std': math.sqrt(variance),
                    'on_time_probability': self._calculate_on_time_probability(
                        mean_completion, variance, customer.due_date
                    )
                }
                
                # Penalty for late completion (expectation over distribution)
                expected_penalty = self._calculate_expected_penalty(
                    mean_completion, variance, customer.due_date, customer.value
                )
                total_penalty += expected_penalty
            
            return {
                'total_expected_penalty': total_penalty,
                'completion_predictions': completion_predictions,
                'avg_on_time_probability': sum(p['on_time_probability'] 
                                             for p in completion_predictions.values()) / len(customers)
            }
        
        def _calculate_on_time_probability(self, mean: float, variance: float, due_date: float) -> float:
            """Calculate probability of on-time completion"""
            if variance <= 0:
                return 1.0 if mean <= due_date else 0.0
            
            # Assume normal distribution
            std = math.sqrt(variance)
            z_score = (due_date - mean) / std
            
            # Approximate normal CDF
            return 0.5 * (1 + math.erf(z_score / math.sqrt(2)))
        
        def _calculate_expected_penalty(self, mean: float, variance: float, 
                                      due_date: float, value: float) -> float:
            """Calculate expected penalty for late completion"""
            # Penalty = max(0, completion_time - due_date) * penalty_rate
            penalty_rate = value * 0.1  # 10% of value per time unit late
            
            if variance <= 0:
                return max(0, mean - due_date) * penalty_rate
            
            # For normal distribution, expected value of max(0, X - d)
            std = math.sqrt(variance)
            z = (due_date - mean) / std
            
            # E[max(0, X - d)] = σ * φ(z) + (μ - d) * Φ(-z)
            # where φ is PDF and Φ is CDF of standard normal
            phi_z = (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z**2)
            cdf_neg_z = 0.5 * (1 + math.erf(-z / math.sqrt(2)))
            
            expected_lateness = std * phi_z + (mean - due_date) * cdf_neg_z
            return max(0, expected_lateness) * penalty_rate
    
    # Demo due date optimization
    production_machines = [
        Machine(0, "CNC", service_rate=0.5, service_variance=0.1),
        Machine(1, "Assembly", service_rate=0.3, service_variance=0.15),
        Machine(2, "Testing", service_rate=0.7, service_variance=0.05),
        Machine(3, "Shipping", service_rate=1.0, service_variance=0.02),
    ]
    
    # Customers with tight due dates
    urgent_customers = []
    for i in range(15):
        arrival = random.randint(0, 20)
        value = random.uniform(100, 500)
        # Tight due dates: 80% of expected completion time
        expected_completion = arrival + 4 * (1/0.5 + 1/0.3 + 1/0.7 + 1/1.0)  # Rough estimate
        due_date = int(arrival + 0.8 * expected_completion)
        
        route = [0, 1, 2, 3]  # Linear production line
        urgent_customers.append(Customer(i, arrival, due_date, value, route=route))
    
    optimizer = DueDateOptimizer(production_machines)
    
    print("Due date optimization scenario:")
    print(f"  {len(urgent_customers)} customers with tight due dates")
    print(f"  Average due date: {sum(c.due_date for c in urgent_customers) / len(urgent_customers):.1f}")
    
    results = optimizer.optimize_due_date_penalties(urgent_customers)
    
    print(f"\nOptimization results:")
    print(f"  Total expected penalty: ${results['total_expected_penalty']:.2f}")
    print(f"  Average on-time probability: {results['avg_on_time_probability']:.1%}")
    
    print(f"\nCustomer completion predictions:")
    for customer_id, pred in list(results['completion_predictions'].items())[:5]:
        print(f"  Customer {customer_id}: "
              f"completion={pred['mean']:.1f}±{pred['std']:.1f}, "
              f"on-time={pred['on_time_probability']:.1%}")


def demo_reinforcement_learning_integration():
    """Integration with reinforcement learning for dynamic queue control"""
    print("\n🧠 RL Integration for Dynamic Queue Control")
    print("=" * 50)
    
    class SimpleRLQueueController:
        """Simplified RL controller demonstrating integration patterns"""
        
        def __init__(self, machines: List[Machine]):
            self.machines = {m.id: m for m in machines}
            self.trees = {m.id: SummaryIntervalTree() for m in machines}
            
            # Simple Q-learning parameters
            self.q_table = defaultdict(lambda: defaultdict(float))
            self.epsilon = 0.1  # Exploration rate
            self.alpha = 0.1    # Learning rate
            self.gamma = 0.9    # Discount factor
            
            # Initialize trees
            for tree in self.trees.values():
                tree.release_interval(0, 1000)
        
        def get_state_features(self, machine_id: int) -> str:
            """Extract state features using tree summaries"""
            stats = self.trees[machine_id].get_availability_stats()
            
            # Discretize features for simple Q-learning
            util_level = int(stats['utilization'] * 5)  # 0-5
            frag_level = int(stats['fragmentation'] * 5)  # 0-5
            capacity_level = int(min(stats['largest_chunk'] / 20, 5))  # 0-5
            
            return f"{machine_id}_{util_level}_{frag_level}_{capacity_level}"
        
        def select_action(self, state: str, available_actions: List[str]) -> str:
            """ε-greedy action selection"""
            if random.random() < self.epsilon:
                return random.choice(available_actions)
            else:
                # Greedy selection
                q_values = [self.q_table[state][action] for action in available_actions]
                best_idx = q_values.index(max(q_values))
                return available_actions[best_idx]
        
        def update_q_value(self, state: str, action: str, reward: float, next_state: str):
            """Q-learning update rule"""
            best_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
            
            current_q = self.q_table[state][action]
            target = reward + self.gamma * best_next_q
            
            self.q_table[state][action] = current_q + self.alpha * (target - current_q)
        
        def simulate_learning(self, episodes: int = 1000) -> Dict:
            """Simulate learning process"""
            episode_rewards = []
            
            for episode in range(episodes):
                episode_reward = 0
                time_step = 0
                
                # Reset environment
                for tree in self.trees.values():
                    tree = SummaryIntervalTree()
                    tree.release_interval(0, 100)
                
                while time_step < 50:  # Episode length
                    # Generate random customer arrival
                    if random.random() < 0.3:  # 30% arrival probability
                        # Select machine to process customer
                        machine_id = random.choice(list(self.machines.keys()))
                        state = self.get_state_features(machine_id)
                        
                        available_actions = ["serve_immediately", "queue_customer", "reject"]
                        action = self.select_action(state, available_actions)
                        
                        # Simulate action outcome
                        if action == "serve_immediately":
                            service_time = random.randint(5, 15)
                            try:
                                window = self.trees[machine_id].find_best_fit(service_time)
                                if window:
                                    self.trees[machine_id].reserve_interval(window[0], window[1])
                                    reward = 10  # Revenue for serving customer
                                else:
                                    reward = -5  # Penalty for failed service
                            except ValueError:
                                reward = -5
                                
                        elif action == "queue_customer":
                            reward = 2  # Small reward for queuing (future potential)
                        else:  # reject
                            reward = -1  # Small penalty for rejection
                        
                        # Observe next state
                        next_state = self.get_state_features(machine_id)
                        
                        # Update Q-value
                        self.update_q_value(state, action, reward, next_state)
                        
                        episode_reward += reward
                    
                    time_step += 1
                
                episode_rewards.append(episode_reward)
                
                if episode % 200 == 0:
                    avg_reward = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
                    print(f"    Episode {episode}: Average reward = {avg_reward:.2f}")
            
            return {
                'episode_rewards': episode_rewards,
                'final_avg_reward': sum(episode_rewards[-100:]) / 100,
                'q_table_size': len(self.q_table),
                'learned_policies': dict(self.q_table)
            }
    
    # Demo RL integration
    machines = [
        Machine(0, "Server_1", service_rate=0.4, service_variance=0.1),
        Machine(1, "Server_2", service_rate=0.6, service_variance=0.15),
    ]
    
    rl_controller = SimpleRLQueueController(machines)
    
    print("RL queue controller setup:")
    for machine in machines:
        print(f"  {machine.name}: rate={machine.service_rate}, variance={machine.service_variance:.2f}")
    
    learning_results = rl_controller.simulate_learning(episodes=1000)
    
    print(f"\nRL learning results:")
    print(f"  Final average reward: {learning_results['final_avg_reward']:.2f}")
    print(f"  Q-table size: {learning_results['q_table_size']:,} state-action pairs")
    
    # Show some learned policies
    print(f"\nSample learned policies:")
    sample_policies = list(learning_results['learned_policies'].items())[:5]
    for state, actions in sample_policies:
        best_action = max(actions.items(), key=lambda x: x[1])
        print(f"  State {state}: Best action = {best_action[0]} (Q={best_action[1]:.2f})")


def main():
    """Run all Bellman iteration demonstrations"""
    print("🔄 Bellman Iteration for Queue Network Optimization")
    print("Dynamic programming and reinforcement learning for stochastic systems")
    print("=" * 75)
    
    random.seed(42)
    if NUMPY_AVAILABLE:
        np.random.seed(42)
    
    demo_simple_queue_network()
    demo_multi_stage_decision_making()
    demo_due_date_optimization()
    demo_reinforcement_learning_integration()
    
    print("\n" + "=" * 75)
    print("✅ Bellman iteration demonstrations complete!")
    print("\n🎯 Key techniques demonstrated:")
    print("  • Bellman optimality equations for queue control")
    print("  • Multi-stage stochastic optimization with dynamic programming")
    print("  • Due date optimization with uncertain completion times")
    print("  • State space compression using tree summary statistics")
    print("  • Integration with reinforcement learning frameworks")
    print("  • Real-time policy evaluation using O(1) tree operations")
    print("  • Stochastic completion time estimation and optimization")


if __name__ == "__main__":
    main()
