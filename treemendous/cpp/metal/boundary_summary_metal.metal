// Metal Compute Shaders for Boundary Summary Operations
// Optimized for Apple Silicon and AMD GPUs on macOS
// Uses Metal Performance Shaders (MPS) framework

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

struct Interval {
    int start;
    int end;
    
    int length() const {
        return end - start;
    }
};

struct SummaryData {
    int total_free_length;
    int largest_interval_length;
    int smallest_interval_length;
    int earliest_start;
    int latest_end;
    int total_gaps;
};

// ============================================================================
// KERNEL: COMPUTE INTERVAL LENGTHS
// ============================================================================

kernel void compute_interval_lengths(
    device const Interval* intervals [[buffer(0)]],
    device int* lengths [[buffer(1)]],
    constant uint& interval_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < interval_count) {
        lengths[gid] = intervals[gid].end - intervals[gid].start;
    }
}

// ============================================================================
// KERNEL: PARALLEL REDUCTION (SUM)
// ============================================================================

kernel void parallel_reduction_sum(
    device const int* input [[buffer(0)]],
    device atomic_int* output [[buffer(1)]],
    constant uint& input_count [[buffer(2)]],
    threadgroup int* shared_data [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint local_size [[threads_per_threadgroup]]
) {
    // Load data into shared memory
    shared_data[tid] = (gid < input_count) ? input[gid] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction in shared memory
    for (uint stride = local_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // First thread writes result
    if (tid == 0) {
        atomic_fetch_add_explicit(output, shared_data[0], memory_order_relaxed);
    }
}

// ============================================================================
// KERNEL: PARALLEL MIN/MAX REDUCTION
// ============================================================================

kernel void parallel_reduction_min_max(
    device const int* input [[buffer(0)]],
    device atomic_int* min_output [[buffer(1)]],
    device atomic_int* max_output [[buffer(2)]],
    constant uint& input_count [[buffer(3)]],
    threadgroup int* shared_min [[threadgroup(0)]],
    threadgroup int* shared_max [[threadgroup(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint local_size [[threads_per_threadgroup]]
) {
    // Initialize with bounds
    shared_min[tid] = (gid < input_count) ? input[gid] : INT_MAX;
    shared_max[tid] = (gid < input_count) ? input[gid] : INT_MIN;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction
    for (uint stride = local_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_min[tid] = min(shared_min[tid], shared_min[tid + stride]);
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // First thread updates global min/max
    if (tid == 0) {
        atomic_fetch_min_explicit(min_output, shared_min[0], memory_order_relaxed);
        atomic_fetch_max_explicit(max_output, shared_max[0], memory_order_relaxed);
    }
}

// ============================================================================
// KERNEL: COMPUTE GAPS BETWEEN INTERVALS
// ============================================================================

kernel void compute_gaps(
    device const Interval* intervals [[buffer(0)]],
    device int* gaps [[buffer(1)]],
    device atomic_int* gap_count [[buffer(2)]],
    constant uint& interval_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < interval_count - 1) {
        int current_end = intervals[gid].end;
        int next_start = intervals[gid + 1].start;
        
        if (next_start > current_end) {
            int gap = next_start - current_end;
            uint idx = atomic_fetch_add_explicit(gap_count, 1, memory_order_relaxed);
            gaps[idx] = gap;
        }
    }
}

// ============================================================================
// KERNEL: FIND BEST FIT (PARALLEL)
// ============================================================================

struct BestFitResult {
    int interval_index;
    int waste;
};

kernel void parallel_best_fit(
    device const Interval* intervals [[buffer(0)]],
    device const int* lengths [[buffer(1)]],
    constant uint& interval_count [[buffer(2)]],
    constant int& required_length [[buffer(3)]],
    constant bool& prefer_early [[buffer(4)]],
    device atomic_int* best_index [[buffer(5)]],
    device atomic_int* best_waste [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < interval_count) {
        int length = lengths[gid];
        
        if (length >= required_length) {
            int waste = length - required_length;
            
            // Atomic compare-and-swap for best fit
            int current_waste = atomic_load_explicit(best_waste, memory_order_relaxed);
            
            while (waste < current_waste) {
                if (atomic_compare_exchange_weak_explicit(
                    best_waste, &current_waste, waste,
                    memory_order_relaxed, memory_order_relaxed)) {
                    atomic_store_explicit(best_index, (int)gid, memory_order_relaxed);
                    break;
                }
            }
            
            // Handle tie-breaking for prefer_early
            if (prefer_early && waste == current_waste) {
                int current_index = atomic_load_explicit(best_index, memory_order_relaxed);
                if ((int)gid < current_index) {
                    atomic_compare_exchange_weak_explicit(
                        best_index, &current_index, (int)gid,
                        memory_order_relaxed, memory_order_relaxed);
                }
            }
        }
    }
}

// ============================================================================
// KERNEL: FIND INTERVAL WITH TARGET LENGTH
// ============================================================================

kernel void find_interval_with_length(
    device const Interval* intervals [[buffer(0)]],
    device const int* lengths [[buffer(1)]],
    constant uint& interval_count [[buffer(2)]],
    constant int& target_length [[buffer(3)]],
    device atomic_int* result_index [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < interval_count) {
        if (lengths[gid] == target_length) {
            // Find first occurrence (minimum index)
            atomic_fetch_min_explicit(result_index, (int)gid, memory_order_relaxed);
        }
    }
}

// ============================================================================
// KERNEL: COMPUTE BOUNDS (EARLIEST/LATEST)
// ============================================================================

kernel void compute_bounds(
    device const Interval* intervals [[buffer(0)]],
    constant uint& interval_count [[buffer(1)]],
    device atomic_int* earliest_start [[buffer(2)]],
    device atomic_int* latest_end [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < interval_count) {
        atomic_fetch_min_explicit(earliest_start, intervals[gid].start, memory_order_relaxed);
        atomic_fetch_max_explicit(latest_end, intervals[gid].end, memory_order_relaxed);
    }
}

// ============================================================================
// KERNEL: FUSED SUMMARY COMPUTATION
// ============================================================================

// Single-pass kernel that computes multiple statistics at once
kernel void compute_summary_fused(
    device const Interval* intervals [[buffer(0)]],
    constant uint& interval_count [[buffer(1)]],
    device atomic_int* total_length [[buffer(2)]],
    device atomic_int* min_length [[buffer(3)]],
    device atomic_int* max_length [[buffer(4)]],
    device atomic_int* earliest_start [[buffer(5)]],
    device atomic_int* latest_end [[buffer(6)]],
    threadgroup int* shared_sum [[threadgroup(0)]],
    threadgroup int* shared_min [[threadgroup(1)]],
    threadgroup int* shared_max [[threadgroup(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint local_size [[threads_per_threadgroup]]
) {
    // Initialize shared memory
    if (gid < interval_count) {
        int length = intervals[gid].end - intervals[gid].start;
        shared_sum[tid] = length;
        shared_min[tid] = length;
        shared_max[tid] = length;
        
        // Also update bounds
        atomic_fetch_min_explicit(earliest_start, intervals[gid].start, memory_order_relaxed);
        atomic_fetch_max_explicit(latest_end, intervals[gid].end, memory_order_relaxed);
    } else {
        shared_sum[tid] = 0;
        shared_min[tid] = INT_MAX;
        shared_max[tid] = INT_MIN;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction for all three metrics
    for (uint stride = local_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_min[tid] = min(shared_min[tid], shared_min[tid + stride]);
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // First thread updates global values
    if (tid == 0) {
        atomic_fetch_add_explicit(total_length, shared_sum[0], memory_order_relaxed);
        atomic_fetch_min_explicit(min_length, shared_min[0], memory_order_relaxed);
        atomic_fetch_max_explicit(max_length, shared_max[0], memory_order_relaxed);
    }
}

// ============================================================================
// KERNEL: SIMD-WIDTH OPTIMIZED REDUCTION
// ============================================================================

// Optimized for Apple Silicon's SIMD width (32 threads)
kernel void simd_reduction_sum(
    device const int* input [[buffer(0)]],
    device atomic_int* output [[buffer(1)]],
    constant uint& input_count [[buffer(2)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint threadgroups [[threadgroups_per_grid]],
    uint gid [[thread_position_in_grid]]
) {
    // Each SIMD group processes elements
    int sum = (gid < input_count) ? input[gid] : 0;
    
    // SIMD-level reduction using shuffle
    sum = simd_sum(sum);
    
    // First lane of each SIMD group writes result
    if (simd_lane_id == 0) {
        atomic_fetch_add_explicit(output, sum, memory_order_relaxed);
    }
}

