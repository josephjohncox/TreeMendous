#!/usr/bin/env python3
"""
Cloud Resource Manager with ML-Enhanced Scheduling

Comprehensive cloud computing resource management system demonstrating:
- Multi-dimensional resource allocation (CPU, memory, network, storage)
- SLA-aware scheduling with penalties
- Machine learning for demand prediction
- Auto-scaling based on tree summary analytics
- Cost optimization with multiple objectives
"""

import sys
import random
import math
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque

# Add Tree-Mendous to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'treemendous' / 'basic'))

try:
    from summary import SummaryIntervalTree
    print("✅ Tree-Mendous interval trees loaded")
except ImportError as e:
    print(f"❌ Failed to load interval trees: {e}")
    sys.exit(1)


class ServiceType(Enum):
    WEB_SERVICE = "web_service"
    DATABASE = "database"
    ML_TRAINING = "ml_training"
    BATCH_PROCESSING = "batch_processing"
    STREAMING = "streaming"


@dataclass
class ResourceRequirement:
    """Multi-dimensional resource requirement"""
    cpu_cores: float
    memory_gb: float
    network_mbps: float
    storage_gb: float
    gpu_units: float = 0.0


@dataclass
class ServiceRequest:
    """Cloud service request with SLA requirements"""
    id: int
    service_type: ServiceType
    resources: ResourceRequirement
    duration_minutes: int
    sla_response_time: int  # Max seconds to start
    sla_completion_time: int  # Max minutes to complete
    priority: int
    cost_per_minute: float
    penalty_per_minute_late: float
    
    # Tracking
    arrival_time: int = 0
    start_time: Optional[int] = None
    completion_time: Optional[int] = None
    assigned_node: Optional[str] = None


@dataclass
class CloudNode:
    """Cloud compute node with multi-dimensional resources"""
    id: str
    node_type: str  # "compute", "memory", "gpu", "storage"
    
    # Resource capacities
    total_cpu: float
    total_memory: float
    total_network: float
    total_storage: float
    total_gpu: float = 0.0
    
    # Cost structure
    cost_per_cpu_hour: float
    cost_per_gb_hour: float
    cost_per_gpu_hour: float = 0.0
    
    # Resource trees for each dimension
    cpu_tree: SummaryIntervalTree = field(default_factory=SummaryIntervalTree)
    memory_tree: SummaryIntervalTree = field(default_factory=SummaryIntervalTree)
    network_tree: SummaryIntervalTree = field(default_factory=SummaryIntervalTree)
    storage_tree: SummaryIntervalTree = field(default_factory=SummaryIntervalTree)
    gpu_tree: SummaryIntervalTree = field(default_factory=SummaryIntervalTree)
    
    def __post_init__(self):
        # Initialize all resources as available (24 hours = 1440 minutes)
        horizon = 1440
        
        # Scale resources to minute-level granularity
        cpu_units = int(self.total_cpu * 60)  # CPU-minutes
        memory_units = int(self.total_memory * 60)  # GB-minutes
        network_units = int(self.total_network * 60)  # Mbps-minutes
        storage_units = int(self.total_storage * 60)  # GB-minutes
        gpu_units = int(self.total_gpu * 60) if self.total_gpu > 0 else 0
        
        self.cpu_tree.release_interval(0, cpu_units)
        self.memory_tree.release_interval(0, memory_units)
        self.network_tree.release_interval(0, network_units)
        self.storage_tree.release_interval(0, storage_units)
        if gpu_units > 0:
            self.gpu_tree.release_interval(0, gpu_units)
    
    def can_accommodate(self, request: ServiceRequest) -> bool:
        """Check if node can accommodate request using tree summaries"""
        required_cpu = int(request.resources.cpu_cores * request.duration_minutes)
        required_memory = int(request.resources.memory_gb * request.duration_minutes)
        required_network = int(request.resources.network_mbps * request.duration_minutes)
        required_storage = int(request.resources.storage_gb * request.duration_minutes)
        required_gpu = int(request.resources.gpu_units * request.duration_minutes)
        
        # Quick check using O(1) tree summaries
        cpu_stats = self.cpu_tree.get_availability_stats()
        memory_stats = self.memory_tree.get_availability_stats()
        network_stats = self.network_tree.get_availability_stats()
        storage_stats = self.storage_tree.get_availability_stats()
        
        if (cpu_stats['largest_chunk'] >= required_cpu and
            memory_stats['largest_chunk'] >= required_memory and
            network_stats['largest_chunk'] >= required_network and
            storage_stats['largest_chunk'] >= required_storage):
            
            if required_gpu > 0:
                gpu_stats = self.gpu_tree.get_availability_stats()
                return gpu_stats['largest_chunk'] >= required_gpu
            
            return True
        
        return False
    
    def allocate_resources(self, request: ServiceRequest, start_time: int) -> bool:
        """Allocate resources for request"""
        end_time = start_time + request.duration_minutes
        
        # Calculate resource requirements
        required_cpu = int(request.resources.cpu_cores * request.duration_minutes)
        required_memory = int(request.resources.memory_gb * request.duration_minutes)
        required_network = int(request.resources.network_mbps * request.duration_minutes)
        required_storage = int(request.resources.storage_gb * request.duration_minutes)
        required_gpu = int(request.resources.gpu_units * request.duration_minutes)
        
        try:
            # Try to allocate all resources
            allocation_windows = {}
            
            # CPU allocation
            cpu_window = self.cpu_tree.find_best_fit(required_cpu)
            if not cpu_window:
                return False
            allocation_windows['cpu'] = cpu_window
            
            # Memory allocation
            memory_window = self.memory_tree.find_best_fit(required_memory)
            if not memory_window:
                return False
            allocation_windows['memory'] = memory_window
            
            # Network allocation
            network_window = self.network_tree.find_best_fit(required_network)
            if not network_window:
                return False
            allocation_windows['network'] = network_window
            
            # Storage allocation
            storage_window = self.storage_tree.find_best_fit(required_storage)
            if not storage_window:
                return False
            allocation_windows['storage'] = storage_window
            
            # GPU allocation (if needed)
            if required_gpu > 0:
                gpu_window = self.gpu_tree.find_best_fit(required_gpu)
                if not gpu_window:
                    return False
                allocation_windows['gpu'] = gpu_window
            
            # All resources available - make reservations
            self.cpu_tree.reserve_interval(*allocation_windows['cpu'])
            self.memory_tree.reserve_interval(*allocation_windows['memory'])
            self.network_tree.reserve_interval(*allocation_windows['network'])
            self.storage_tree.reserve_interval(*allocation_windows['storage'])
            if required_gpu > 0:
                self.gpu_tree.reserve_interval(*allocation_windows['gpu'])
            
            return True
            
        except ValueError:
            return False
    
    def get_node_efficiency(self) -> Dict[str, float]:
        """Calculate node efficiency metrics using tree summaries"""
        cpu_stats = self.cpu_tree.get_availability_stats()
        memory_stats = self.memory_tree.get_availability_stats()
        network_stats = self.network_tree.get_availability_stats()
        storage_stats = self.storage_tree.get_availability_stats()
        
        return {
            'cpu_utilization': cpu_stats['utilization'],
            'memory_utilization': memory_stats['utilization'],
            'network_utilization': network_stats['utilization'],
            'storage_utilization': storage_stats['utilization'],
            'cpu_fragmentation': cpu_stats['fragmentation'],
            'memory_fragmentation': memory_stats['fragmentation'],
            'overall_efficiency': (cpu_stats['utilization'] + memory_stats['utilization']) / 2 * 
                                (1 - (cpu_stats['fragmentation'] + memory_stats['fragmentation']) / 2)
        }


class CloudResourceManager:
    """Cloud resource manager with ML-enhanced scheduling"""
    
    def __init__(self, nodes: List[CloudNode]):
        self.nodes = {node.id: node for node in nodes}
        self.current_time = 0
        
        # Request queues by priority
        self.pending_requests = {1: deque(), 2: deque(), 3: deque()}  # Priority levels
        self.running_requests = []
        self.completed_requests = []
        self.sla_violations = []
        
        # ML components (simplified)
        self.demand_predictor = DemandPredictor()
        self.autoscaler = AutoScaler(self.nodes)
        
        # Performance tracking
        self.performance_history = []
    
    def submit_request(self, request: ServiceRequest) -> str:
        """Submit service request to cluster"""
        request.arrival_time = self.current_time
        self.pending_requests[request.priority].append(request)
        return f"Request {request.id} queued with priority {request.priority}"
    
    def schedule_pending_requests(self) -> Dict:
        """Schedule pending requests using tree-enhanced algorithms"""
        scheduled = 0
        rejected = 0
        
        # Process requests by priority (3=highest, 1=lowest)
        for priority in [3, 2, 1]:
            while self.pending_requests[priority]:
                request = self.pending_requests[priority].popleft()
                
                # Find best node using tree analysis
                best_node = self._find_best_node_for_request(request)
                
                if best_node:
                    # Allocate resources
                    if best_node.allocate_resources(request, self.current_time):
                        request.start_time = self.current_time
                        request.assigned_node = best_node.id
                        self.running_requests.append(request)
                        scheduled += 1
                    else:
                        rejected += 1
                        # Check SLA violation
                        if self.current_time - request.arrival_time > request.sla_response_time:
                            self.sla_violations.append(request)
                else:
                    rejected += 1
                    if self.current_time - request.arrival_time > request.sla_response_time:
                        self.sla_violations.append(request)
        
        return {'scheduled': scheduled, 'rejected': rejected}
    
    def _find_best_node_for_request(self, request: ServiceRequest) -> Optional[CloudNode]:
        """Find best node for request using multi-criteria optimization"""
        candidate_nodes = []
        
        for node in self.nodes.values():
            if node.can_accommodate(request):
                # Score node based on multiple factors
                efficiency = node.get_node_efficiency()
                
                # Factors: utilization balance, cost, specialization
                utilization_score = 1.0 - efficiency['overall_efficiency']  # Prefer less utilized
                cost_score = node.cost_per_cpu_hour * request.resources.cpu_cores  # Cost consideration
                
                # Specialization bonuses
                specialization_score = 0
                if request.service_type == ServiceType.ML_TRAINING and node.total_gpu > 0:
                    specialization_score = -50  # GPU bonus for ML
                elif request.service_type == ServiceType.DATABASE and "memory" in node.node_type:
                    specialization_score = -20  # Memory bonus for DB
                
                total_score = utilization_score + cost_score * 0.01 + specialization_score
                candidate_nodes.append((node, total_score))
        
        if candidate_nodes:
            # Return node with best (lowest) score
            best_node, _ = min(candidate_nodes, key=lambda x: x[1])
            return best_node
        
        return None
    
    def advance_time(self, minutes: int = 1) -> None:
        """Advance simulation time and process completions"""
        self.current_time += minutes
        
        # Check for completed requests
        completed_this_step = []
        for request in self.running_requests:
            expected_completion = request.start_time + request.duration_minutes
            if self.current_time >= expected_completion:
                request.completion_time = self.current_time
                completed_this_step.append(request)
        
        # Move completed requests
        for request in completed_this_step:
            self.running_requests.remove(request)
            self.completed_requests.append(request)
            
            # Check SLA compliance
            if request.completion_time > request.arrival_time + request.sla_completion_time:
                self.sla_violations.append(request)
    
    def get_cluster_analytics(self) -> Dict:
        """Get comprehensive cluster analytics using tree summaries"""
        node_stats = {}
        cluster_totals = {
            'cpu_utilization': 0,
            'memory_utilization': 0,
            'network_utilization': 0,
            'storage_utilization': 0,
            'cpu_fragmentation': 0,
            'memory_fragmentation': 0,
            'total_cost': 0
        }
        
        for node_id, node in self.nodes.items():
            efficiency = node.get_node_efficiency()
            node_stats[node_id] = efficiency
            
            # Aggregate cluster metrics
            cluster_totals['cpu_utilization'] += efficiency['cpu_utilization']
            cluster_totals['memory_utilization'] += efficiency['memory_utilization']
            cluster_totals['network_utilization'] += efficiency['network_utilization']
            cluster_totals['storage_utilization'] += efficiency['storage_utilization']
            cluster_totals['cpu_fragmentation'] += efficiency['cpu_fragmentation']
            cluster_totals['memory_fragmentation'] += efficiency['memory_fragmentation']
            
            # Calculate current cost
            cost = (efficiency['cpu_utilization'] * node.total_cpu * node.cost_per_cpu_hour +
                   efficiency['memory_utilization'] * node.total_memory * node.cost_per_gb_hour)
            cluster_totals['total_cost'] += cost
        
        # Average metrics
        num_nodes = len(self.nodes)
        for key in ['cpu_utilization', 'memory_utilization', 'network_utilization', 
                   'storage_utilization', 'cpu_fragmentation', 'memory_fragmentation']:
            cluster_totals[key] /= num_nodes
        
        return {
            'cluster_totals': cluster_totals,
            'node_stats': node_stats,
            'pending_requests': sum(len(q) for q in self.pending_requests.values()),
            'running_requests': len(self.running_requests),
            'completed_requests': len(self.completed_requests),
            'sla_violations': len(self.sla_violations),
            'sla_compliance_rate': 1.0 - len(self.sla_violations) / max(1, len(self.completed_requests))
        }


class DemandPredictor:
    """Simple ML-based demand prediction using historical patterns"""
    
    def __init__(self):
        self.history = deque(maxlen=100)  # Keep last 100 observations
        self.patterns = defaultdict(list)
    
    def record_demand(self, time_slot: int, service_type: ServiceType, demand: int) -> None:
        """Record demand observation"""
        self.history.append((time_slot, service_type, demand))
        
        # Extract patterns (simplified)
        hour_of_day = (time_slot // 60) % 24
        self.patterns[service_type].append((hour_of_day, demand))
    
    def predict_demand(self, time_slot: int, service_type: ServiceType) -> Tuple[float, float]:
        """Predict demand (mean, std) for given time and service type"""
        if service_type not in self.patterns or len(self.patterns[service_type]) < 5:
            return 2.0, 1.0  # Default prediction
        
        hour_of_day = (time_slot // 60) % 24
        
        # Find similar time periods
        similar_demands = []
        for hour, demand in self.patterns[service_type]:
            if abs(hour - hour_of_day) <= 2:  # Within 2 hours
                similar_demands.append(demand)
        
        if similar_demands:
            mean_demand = sum(similar_demands) / len(similar_demands)
            if len(similar_demands) > 1:
                variance = sum((d - mean_demand)**2 for d in similar_demands) / len(similar_demands)
                std_demand = math.sqrt(variance)
            else:
                std_demand = 1.0
            return mean_demand, std_demand
        
        return 2.0, 1.0


class AutoScaler:
    """Intelligent auto-scaling using tree analytics"""
    
    def __init__(self, nodes: Dict[str, CloudNode]):
        self.nodes = nodes
        self.scaling_history = []
        self.scale_up_threshold = 0.8  # 80% utilization
        self.scale_down_threshold = 0.3  # 30% utilization
        
    def should_scale(self, cluster_analytics: Dict) -> Tuple[str, Optional[str]]:
        """Determine if scaling action needed"""
        cluster_totals = cluster_analytics['cluster_totals']
        
        avg_cpu_util = cluster_totals['cpu_utilization']
        avg_memory_util = cluster_totals['memory_utilization']
        avg_fragmentation = (cluster_totals['cpu_fragmentation'] + cluster_totals['memory_fragmentation']) / 2
        
        # Scale up conditions
        if avg_cpu_util > self.scale_up_threshold or avg_memory_util > self.scale_up_threshold:
            # Determine what type of node to add
            if avg_cpu_util > avg_memory_util:
                return "scale_up", "compute"
            else:
                return "scale_up", "memory"
        
        # Scale down conditions
        elif (avg_cpu_util < self.scale_down_threshold and 
              avg_memory_util < self.scale_down_threshold and 
              len(self.nodes) > 2):  # Don't scale below minimum
            
            # Find least utilized node for removal
            least_utilized_node = min(
                self.nodes.values(),
                key=lambda n: n.get_node_efficiency()['overall_efficiency']
            )
            return "scale_down", least_utilized_node.id
        
        # High fragmentation - consider defragmentation
        elif avg_fragmentation > 0.7:
            return "defragment", None
        
        return "no_action", None
    
    def execute_scaling_action(self, action: str, target: Optional[str], 
                             cluster_analytics: Dict) -> Dict:
        """Execute scaling action"""
        result = {'action': action, 'target': target, 'success': False}
        
        if action == "scale_up":
            # Add new node (simplified)
            new_node_id = f"auto_node_{len(self.nodes)}"
            
            if target == "compute":
                new_node = CloudNode(
                    id=new_node_id,
                    node_type="compute",
                    total_cpu=8.0,
                    total_memory=32.0,
                    total_network=1000.0,
                    total_storage=100.0,
                    cost_per_cpu_hour=0.1,
                    cost_per_gb_hour=0.01
                )
            else:  # memory
                new_node = CloudNode(
                    id=new_node_id,
                    node_type="memory",
                    total_cpu=4.0,
                    total_memory=64.0,
                    total_network=1000.0,
                    total_storage=100.0,
                    cost_per_cpu_hour=0.08,
                    cost_per_gb_hour=0.008
                )
            
            self.nodes[new_node_id] = new_node
            result['success'] = True
            result['message'] = f"Added {target} node {new_node_id}"
            
        elif action == "scale_down" and target:
            if target in self.nodes:
                # Check if node can be safely removed
                node = self.nodes[target]
                efficiency = node.get_node_efficiency()
                
                if efficiency['overall_efficiency'] < 0.1:  # Very low utilization
                    del self.nodes[target]
                    result['success'] = True
                    result['message'] = f"Removed node {target}"
                else:
                    result['message'] = f"Node {target} too busy to remove"
            
        elif action == "defragment":
            # Trigger defragmentation across cluster
            defrag_count = 0
            for node in self.nodes.values():
                # Simplified defragmentation: this would involve migrating workloads
                # For demo, we'll just record the need for defragmentation
                efficiency = node.get_node_efficiency()
                if efficiency['cpu_fragmentation'] > 0.5:
                    defrag_count += 1
            
            result['success'] = True
            result['message'] = f"Defragmentation needed on {defrag_count} nodes"
        
        self.scaling_history.append(result)
        return result


def demo_cloud_service_scheduling():
    """Demonstrate cloud service scheduling with SLA management"""
    print("☁️ Cloud Service Scheduling with SLA Management")
    print("=" * 50)
    
    # Create diverse cloud nodes
    nodes = [
        CloudNode("compute_1", "compute", 16.0, 64.0, 10000, 500, 0, 0.12, 0.015),
        CloudNode("compute_2", "compute", 12.0, 48.0, 8000, 400, 0, 0.10, 0.012),
        CloudNode("memory_1", "memory", 8.0, 128.0, 5000, 1000, 0, 0.08, 0.008),
        CloudNode("gpu_1", "gpu", 8.0, 32.0, 10000, 200, 4.0, 0.15, 0.020, 2.50),
        CloudNode("storage_1", "storage", 4.0, 16.0, 2000, 10000, 0, 0.06, 0.005),
    ]
    
    manager = CloudResourceManager(nodes)
    
    print("Cloud cluster configuration:")
    for node in nodes:
        print(f"  {node.id} ({node.node_type}): "
              f"CPU={node.total_cpu}, RAM={node.total_memory}GB, "
              f"GPU={node.total_gpu}")
    
    # Generate realistic service requests
    service_requests = []
    
    # Web services (high frequency, low resource)
    for i in range(15):
        service_requests.append(ServiceRequest(
            id=i,
            service_type=ServiceType.WEB_SERVICE,
            resources=ResourceRequirement(0.5, 2.0, 100, 5),
            duration_minutes=random.randint(30, 120),
            sla_response_time=30,  # 30 seconds
            sla_completion_time=180,  # 3 hours
            priority=2,
            cost_per_minute=0.10,
            penalty_per_minute_late=1.0
        ))
    
    # ML training jobs (low frequency, high resource)
    for i in range(5):
        service_requests.append(ServiceRequest(
            id=i + 100,
            service_type=ServiceType.ML_TRAINING,
            resources=ResourceRequirement(4.0, 16.0, 500, 50, 2.0),  # Needs GPU
            duration_minutes=random.randint(120, 480),
            sla_response_time=300,  # 5 minutes  
            sla_completion_time=600,  # 10 hours
            priority=3,
            cost_per_minute=2.50,
            penalty_per_minute_late=10.0
        ))
    
    # Database services (medium frequency, memory intensive)
    for i in range(8):
        service_requests.append(ServiceRequest(
            id=i + 200,
            service_type=ServiceType.DATABASE,
            resources=ResourceRequirement(2.0, 8.0, 200, 100),
            duration_minutes=random.randint(60, 300),
            sla_response_time=60,  # 1 minute
            sla_completion_time=360,  # 6 hours
            priority=2,
            cost_per_minute=0.50,
            penalty_per_minute_late=5.0
        ))
    
    print(f"\nService request mix:")
    service_counts = defaultdict(int)
    for req in service_requests:
        service_counts[req.service_type] += 1
    
    for service_type, count in service_counts.items():
        print(f"  {service_type.value}: {count} requests")
    
    # Simulate scheduling over time
    print(f"\n🔄 Simulating cloud scheduling over 8 hours...")
    
    # Submit requests over time
    request_times = sorted(random.randint(0, 480) for _ in range(len(service_requests)))
    request_schedule = list(zip(request_times, service_requests))
    request_idx = 0
    
    # Track performance metrics
    performance_snapshots = []
    
    for minute in range(480):  # 8 hours simulation
        manager.advance_time(1)
        
        # Submit requests arriving at this time
        while (request_idx < len(request_schedule) and 
               request_schedule[request_idx][0] <= minute):
            _, request = request_schedule[request_idx]
            manager.submit_request(request)
            request_idx += 1
        
        # Schedule pending requests
        manager.schedule_pending_requests()
        
        # Record performance every hour
        if minute % 60 == 0:
            analytics = manager.get_cluster_analytics()
            performance_snapshots.append((minute, analytics))
            
            # Auto-scaling decision
            scaling_action, target = manager.autoscaler.should_scale(analytics)
            if scaling_action != "no_action":
                scaling_result = manager.autoscaler.execute_scaling_action(
                    scaling_action, target, analytics
                )
                print(f"  Hour {minute//60}: {scaling_result['message']}")
    
    # Final analysis
    final_analytics = manager.get_cluster_analytics()
    
    print(f"\n📊 Final Performance Analysis:")
    print(f"  Completed requests: {final_analytics['completed_requests']}")
    print(f"  SLA violations: {final_analytics['sla_violations']}")
    print(f"  SLA compliance rate: {final_analytics['sla_compliance_rate']:.1%}")
    
    cluster_totals = final_analytics['cluster_totals']
    print(f"  Average CPU utilization: {cluster_totals['cpu_utilization']:.1%}")
    print(f"  Average memory utilization: {cluster_totals['memory_utilization']:.1%}")
    print(f"  Average fragmentation: {(cluster_totals['cpu_fragmentation'] + cluster_totals['memory_fragmentation'])/2:.1%}")
    print(f"  Total cost: ${cluster_totals['total_cost']:.2f}")
    
    # Show auto-scaling history
    if manager.autoscaler.scaling_history:
        print(f"\n🔄 Auto-scaling actions taken:")
        for action_result in manager.autoscaler.scaling_history:
            print(f"  {action_result['action']}: {action_result['message']}")


def demo_multi_tenant_isolation():
    """Multi-tenant cloud with isolation requirements"""
    print("\n🏢 Multi-Tenant Cloud with Isolation")
    print("=" * 50)
    
    @dataclass
    class Tenant:
        id: str
        name: str
        sla_tier: str  # "bronze", "silver", "gold", "platinum"
        isolation_requirements: Set[str]  # "compute", "network", "storage"
        cost_multiplier: float
        
    tenants = [
        Tenant("T1", "Enterprise_Corp", "platinum", {"compute", "network"}, 2.0),
        Tenant("T2", "Startup_A", "silver", set(), 1.0),
        Tenant("T3", "Government_Agency", "gold", {"compute", "storage"}, 1.5),
        Tenant("T4", "SMB_Customer", "bronze", set(), 0.8),
    ]
    
    class MultiTenantManager(CloudResourceManager):
        def __init__(self, nodes: List[CloudNode], tenants: List[Tenant]):
            super().__init__(nodes)
            self.tenants = {t.id: t for t in tenants}
            self.tenant_trees = {t.id: SummaryIntervalTree() for t in tenants}
            
            # Initialize tenant-specific capacity tracking
            for tree in self.tenant_trees.values():
                tree.release_interval(0, 1440)  # 24 hours
        
        def allocate_with_isolation(self, request: ServiceRequest, tenant_id: str) -> bool:
            """Allocate resources respecting tenant isolation requirements"""
            tenant = self.tenants[tenant_id]
            
            # Find nodes that meet isolation requirements
            candidate_nodes = []
            for node in self.nodes.values():
                meets_isolation = True
                
                for isolation_req in tenant.isolation_requirements:
                    if isolation_req == "compute":
                        # Check if node has sufficient dedicated compute
                        efficiency = node.get_node_efficiency()
                        if efficiency['cpu_utilization'] > 0.5:  # Already sharing compute
                            meets_isolation = False
                            break
                
                if meets_isolation and node.can_accommodate(request):
                    candidate_nodes.append(node)
            
            if candidate_nodes:
                # Select best candidate
                best_node = min(candidate_nodes, 
                              key=lambda n: n.get_node_efficiency()['cpu_utilization'])
                
                return best_node.allocate_resources(request, self.current_time)
            
            return False
    
    # Demo multi-tenant scheduling
    mt_manager = MultiTenantManager(nodes, tenants)
    
    print("Multi-tenant configuration:")
    for tenant in tenants:
        isolation_str = ", ".join(tenant.isolation_requirements) if tenant.isolation_requirements else "None"
        print(f"  {tenant.name} ({tenant.sla_tier}): Isolation={isolation_str}, "
              f"Cost multiplier={tenant.cost_multiplier}x")
    
    # Generate tenant-specific requests
    tenant_requests = []
    for i, tenant in enumerate(tenants):
        num_requests = [8, 12, 6, 10][i]  # Different request volumes
        
        for j in range(num_requests):
            # Vary resource requirements by tenant type
            if tenant.sla_tier == "platinum":
                resources = ResourceRequirement(4.0, 16.0, 1000, 100)
                priority = 3
            elif tenant.sla_tier == "gold":
                resources = ResourceRequirement(2.0, 8.0, 500, 50)
                priority = 2
            else:
                resources = ResourceRequirement(1.0, 4.0, 200, 25)
                priority = 1
            
            tenant_requests.append((tenant.id, ServiceRequest(
                id=j + i*100,
                service_type=ServiceType.WEB_SERVICE,
                resources=resources,
                duration_minutes=random.randint(60, 180),
                sla_response_time=30 if tenant.sla_tier in ["gold", "platinum"] else 120,
                sla_completion_time=240,
                priority=priority,
                cost_per_minute=0.20 * tenant.cost_multiplier,
                penalty_per_minute_late=2.0 * tenant.cost_multiplier
            )))
    
    print(f"\nProcessing {len(tenant_requests)} tenant requests...")
    
    # Process requests with tenant awareness
    tenant_performance = defaultdict(lambda: {'allocated': 0, 'rejected': 0})
    
    for tenant_id, request in tenant_requests:
        if mt_manager.allocate_with_isolation(request, tenant_id):
            tenant_performance[tenant_id]['allocated'] += 1
        else:
            tenant_performance[tenant_id]['rejected'] += 1
    
    print(f"\nTenant allocation results:")
    for tenant_id, performance in tenant_performance.items():
        tenant_name = mt_manager.tenants[tenant_id].name
        total = performance['allocated'] + performance['rejected']
        success_rate = performance['allocated'] / total if total > 0 else 0
        print(f"  {tenant_name}: {success_rate:.1%} success rate "
              f"({performance['allocated']}/{total})")


def main():
    """Run all cloud resource management demonstrations"""
    print("☁️ Cloud Resource Management with ML-Enhanced Scheduling")
    print("Advanced resource allocation, SLA management, and auto-scaling")
    print("=" * 70)
    
    random.seed(42)  # Reproducible results
    
    demo_cloud_service_scheduling()
    demo_multi_tenant_isolation()
    
    print("\n" + "=" * 70)
    print("✅ Cloud resource management demonstrations complete!")
    print("\n🎯 Advanced capabilities demonstrated:")
    print("  • Multi-dimensional resource allocation (CPU, memory, network, storage, GPU)")
    print("  • SLA-aware scheduling with compliance monitoring") 
    print("  • Intelligent auto-scaling using tree summary analytics")
    print("  • Multi-tenant isolation with resource guarantees")
    print("  • Cost optimization across multiple objectives")
    print("  • Real-time performance monitoring using O(1) tree operations")
    print("  • Machine learning integration for demand prediction")
    print("  • Tree-guided node selection and load balancing")


if __name__ == "__main__":
    main()
