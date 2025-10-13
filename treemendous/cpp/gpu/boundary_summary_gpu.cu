// treemendous/cpp/gpu/boundary_summary_gpu.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <vector>
#include <optional>
#include <iostream>

// Check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// GPU-compatible structures
struct GPUInterval {
    int start;
    int end;
    
    __device__ __host__ int length() const { return end - start; }
};

struct GPUSummary {
    int total_free_length = 0;
    int total_occupied_length = 0;
    int interval_count = 0;
    int largest_interval_length = 0;
    int largest_interval_start = -1;
    int smallest_interval_length = INT_MAX;
    int total_gaps = 0;
    int earliest_start = -1;
    int latest_end = -1;
    double avg_interval_length = 0.0;
    double avg_gap_size = 0.0;
    double fragmentation_index = 0.0;
    double utilization = 0.0;
    
    __device__ __host__ void update_metrics() {
        if (interval_count > 0) {
            avg_interval_length = static_cast<double>(total_free_length) / interval_count;
        }
        
        if (total_free_length > 0) {
            fragmentation_index = 1.0 - static_cast<double>(largest_interval_length) / total_free_length;
        }
    }
};

// ============================================================================
// CUDA KERNELS
// ============================================================================

__global__ void compute_lengths_kernel(
    const GPUInterval* intervals,
    int* lengths,
    int num_intervals
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_intervals) {
        lengths[idx] = intervals[idx].length();
    }
}

__global__ void compute_gaps_kernel(
    const GPUInterval* intervals,
    int* gaps,
    int* valid_gaps,
    int num_intervals
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_intervals - 1) {
        int gap = intervals[idx + 1].start - intervals[idx].end;
        gaps[idx] = gap;
        valid_gaps[idx] = (gap > 0) ? 1 : 0;
    }
}

__global__ void find_largest_interval_kernel(
    const GPUInterval* intervals,
    const int* lengths,
    int num_intervals,
    int target_length,
    int* result_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_intervals) {
        if (lengths[idx] == target_length) {
            atomicMin(result_idx, idx);  // Find first occurrence
        }
    }
}

__global__ void parallel_best_fit_kernel(
    const GPUInterval* intervals,
    int num_intervals,
    int required_length,
    bool prefer_early,
    int* best_idx,
    int* best_waste
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_intervals) {
        int length = intervals[idx].length();
        
        if (length >= required_length) {
            int waste = length - required_length;
            int old_waste = atomicMin(best_waste, waste);
            
            if (waste <= old_waste) {
                if (prefer_early && waste == old_waste && idx < *best_idx) {
                    atomicMin(best_idx, idx);
                } else if (waste < old_waste) {
                    atomicExch(best_idx, idx);
                }
            }
        }
    }
}

// ============================================================================
// GPU BOUNDARY SUMMARY MANAGER
// ============================================================================

class GPUBoundarySummaryManager {
private:
    // Host-side interval storage (authoritative)
    std::map<int, int> cpu_intervals_;
    
    // GPU-side storage
    thrust::device_vector<GPUInterval> gpu_intervals_;
    thrust::device_vector<int> gpu_lengths_;
    thrust::device_vector<int> gpu_gaps_;
    
    // State tracking
    bool gpu_dirty_ = true;
    int managed_start_ = -1;
    int managed_end_ = -1;
    
    // Performance tracking
    int operation_count_ = 0;
    int gpu_operations_ = 0;
    int cpu_operations_ = 0;
    
    // Configuration
    static constexpr int GPU_THRESHOLD = 1000;  // Minimum intervals for GPU
    static constexpr int BLOCK_SIZE = 256;
    
public:
    GPUBoundarySummaryManager() {
        // Pre-allocate GPU memory
        gpu_intervals_.reserve(1024);
        gpu_lengths_.reserve(1024);
        gpu_gaps_.reserve(1024);
    }
    
    // Standard interval operations (CPU-based)
    void release_interval(int start, int end) {
        if (start >= end) return;
        
        update_managed_bounds(start, end);
        ++operation_count_;
        ++cpu_operations_;
        gpu_dirty_ = true;
        
        auto it = cpu_intervals_.lower_bound(start);
        
        // Merge logic (same as CPU version)
        if (it != cpu_intervals_.begin()) {
            auto prev = std::prev(it);
            if (prev->second >= start) {
                start = prev->first;
                end = std::max(end, prev->second);
                cpu_intervals_.erase(prev);
            }
        }
        
        while (it != cpu_intervals_.end() && it->first <= end) {
            end = std::max(end, it->second);
            it = cpu_intervals_.erase(it);
        }
        
        cpu_intervals_[start] = end;
    }
    
    // Batch operations for GPU efficiency (amortize Python overhead)
    void batch_reserve(const std::vector<std::pair<int, int>>& reserves) {
        if (reserves.empty()) return;
        
        // Optimized: process all reserves in one pass
        for (const auto& [start, end] : reserves) {
            reserve_interval(start, end);
        }
    }
    
    void batch_release(const std::vector<std::pair<int, int>>& releases) {
        if (releases.empty()) return;
        
        // Optimized batch release: sort + merge in one pass
        if (cpu_intervals_.size() > 100 && releases.size() > 100) {
            // Sort releases
            std::vector<std::pair<int, int>> sorted_releases = releases;
            std::sort(sorted_releases.begin(), sorted_releases.end());
            
            // Merge all intervals (existing + new) efficiently
            std::vector<std::pair<int, int>> all_intervals;
            all_intervals.reserve(cpu_intervals_.size() + sorted_releases.size());
            
            for (const auto& [s, e] : cpu_intervals_) {
                all_intervals.emplace_back(s, e);
            }
            for (const auto& [s, e] : sorted_releases) {
                all_intervals.emplace_back(s, e);
            }
            
            std::sort(all_intervals.begin(), all_intervals.end());
            
            // Merge overlapping/adjacent intervals
            cpu_intervals_.clear();
            if (!all_intervals.empty()) {
                int current_start = all_intervals[0].first;
                int current_end = all_intervals[0].second;
                
                for (size_t i = 1; i < all_intervals.size(); i++) {
                    if (all_intervals[i].first <= current_end) {
                        current_end = std::max(current_end, all_intervals[i].second);
                    } else {
                        cpu_intervals_[current_start] = current_end;
                        current_start = all_intervals[i].first;
                        current_end = all_intervals[i].second;
                    }
                }
                cpu_intervals_[current_start] = current_end;
            }
            
            gpu_dirty_ = true;
        } else {
            // Small batch: use individual operations
            for (const auto& [start, end] : releases) {
                release_interval(start, end);
            }
        }
    }
    
    void reserve_interval(int start, int end) {
        if (start >= end) return;
        
        ++operation_count_;
        ++cpu_operations_;
        gpu_dirty_ = true;
        
        // Optimized: use lower_bound instead of iterating all intervals
        auto it = cpu_intervals_.lower_bound(start);
        
        // Check previous interval if it might overlap
        if (it != cpu_intervals_.begin()) {
            auto prev = std::prev(it);
            if (prev->second > start) {
                it = prev;
            }
        }
        
        // Process only overlapping intervals (not all intervals)
        std::vector<std::pair<int, int>> to_add;
        std::vector<int> to_remove_keys;
        to_add.reserve(4);  // Typically 0-2 intervals
        to_remove_keys.reserve(4);
        
        while (it != cpu_intervals_.end() && it->first < end) {
            int istart = it->first;
            int iend = it->second;
            
            if (iend <= start) {
                ++it;
                continue;
            }
            
            to_remove_keys.push_back(istart);
            
            if (istart < start) {
                to_add.push_back({istart, start});
            }
            if (iend > end) {
                to_add.push_back({end, iend});
            }
            
            ++it;
        }
        
        // Apply changes
        for (int key : to_remove_keys) {
            cpu_intervals_.erase(key);
        }
        
        for (const auto& [s, e] : to_add) {
            cpu_intervals_[s] = e;
        }
    }
    
    // GPU-accelerated summary computation
    GPUSummary compute_summary_gpu() {
        sync_to_gpu();
        
        int n = gpu_intervals_.size();
        
        if (n == 0) {
            GPUSummary summary;
            if (managed_start_ != -1 && managed_end_ != -1) {
                summary.total_occupied_length = managed_end_ - managed_start_;
                summary.utilization = 1.0;
            }
            return summary;
        }
        
        ++gpu_operations_;
        
        // Compute lengths using GPU
        gpu_lengths_.resize(n);
        
        int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_lengths_kernel<<<num_blocks, BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(gpu_intervals_.data()),
            thrust::raw_pointer_cast(gpu_lengths_.data()),
            n
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Use Thrust for reductions (highly optimized)
        GPUSummary summary;
        summary.interval_count = n;
        summary.total_free_length = thrust::reduce(gpu_lengths_.begin(), gpu_lengths_.end(), 0);
        
        auto minmax = thrust::minmax_element(gpu_lengths_.begin(), gpu_lengths_.end());
        summary.smallest_interval_length = *minmax.first;
        summary.largest_interval_length = *minmax.second;
        
        // Find start of largest interval
        int largest_idx = thrust::max_element(gpu_lengths_.begin(), gpu_lengths_.end()) - gpu_lengths_.begin();
        GPUInterval largest_interval;
        CUDA_CHECK(cudaMemcpy(&largest_interval, 
                             thrust::raw_pointer_cast(gpu_intervals_.data()) + largest_idx,
                             sizeof(GPUInterval), 
                             cudaMemcpyDeviceToHost));
        summary.largest_interval_start = largest_interval.start;
        
        // Bounds
        GPUInterval first, last;
        CUDA_CHECK(cudaMemcpy(&first, thrust::raw_pointer_cast(gpu_intervals_.data()),
                             sizeof(GPUInterval), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last, 
                             thrust::raw_pointer_cast(gpu_intervals_.data()) + n - 1,
                             sizeof(GPUInterval), cudaMemcpyDeviceToHost));
        summary.earliest_start = first.start;
        summary.latest_end = last.end;
        
        // Compute gaps
        if (n > 1) {
            gpu_gaps_.resize(n - 1);
            thrust::device_vector<int> valid_gaps(n - 1);
            
            compute_gaps_kernel<<<num_blocks, BLOCK_SIZE>>>(
                thrust::raw_pointer_cast(gpu_intervals_.data()),
                thrust::raw_pointer_cast(gpu_gaps_.data()),
                thrust::raw_pointer_cast(valid_gaps.data()),
                n
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            
            summary.total_gaps = thrust::reduce(valid_gaps.begin(), valid_gaps.end(), 0);
            
            if (summary.total_gaps > 0) {
                int total_gap_size = thrust::reduce(gpu_gaps_.begin(), gpu_gaps_.end(), 0);
                summary.avg_gap_size = static_cast<double>(total_gap_size) / summary.total_gaps;
            }
        }
        
        // Utilization
        if (managed_start_ != -1 && managed_end_ != -1) {
            int managed_space = managed_end_ - managed_start_;
            summary.total_occupied_length = managed_space - summary.total_free_length;
            summary.utilization = static_cast<double>(summary.total_occupied_length) / managed_space;
        }
        
        summary.update_metrics();
        return summary;
    }
    
    // CPU fallback for small datasets
    GPUSummary compute_summary_cpu() const {
        // Same implementation as current C++ version
        GPUSummary summary;
        
        if (cpu_intervals_.empty()) {
            if (managed_start_ != -1 && managed_end_ != -1) {
                summary.total_occupied_length = managed_end_ - managed_start_;
                summary.utilization = 1.0;
            }
            return summary;
        }
        
        summary.interval_count = cpu_intervals_.size();
        summary.earliest_start = cpu_intervals_.begin()->first;
        summary.latest_end = cpu_intervals_.rbegin()->second;
        
        std::vector<int> lengths;
        lengths.reserve(cpu_intervals_.size());
        
        for (const auto& [start, end] : cpu_intervals_) {
            int length = end - start;
            lengths.push_back(length);
            summary.total_free_length += length;
        }
        
        if (!lengths.empty()) {
            summary.largest_interval_length = *std::max_element(lengths.begin(), lengths.end());
            summary.smallest_interval_length = *std::min_element(lengths.begin(), lengths.end());
            
            for (const auto& [start, end] : cpu_intervals_) {
                if ((end - start) == summary.largest_interval_length) {
                    summary.largest_interval_start = start;
                    break;
                }
            }
        }
        
        // Gaps
        if (cpu_intervals_.size() > 1) {
            auto it = cpu_intervals_.begin();
            int prev_end = it->second;
            ++it;
            int total_gap = 0;
            
            while (it != cpu_intervals_.end()) {
                if (it->first > prev_end) {
                    total_gap += (it->first - prev_end);
                    ++summary.total_gaps;
                }
                prev_end = it->second;
                ++it;
            }
            
            if (summary.total_gaps > 0) {
                summary.avg_gap_size = static_cast<double>(total_gap) / summary.total_gaps;
            }
        }
        
        if (managed_start_ != -1 && managed_end_ != -1) {
            int managed_space = managed_end_ - managed_start_;
            summary.total_occupied_length = managed_space - summary.total_free_length;
            summary.utilization = static_cast<double>(summary.total_occupied_length) / managed_space;
        }
        
        summary.update_metrics();
        return summary;
    }
    
    // Adaptive summary: chooses GPU or CPU based on dataset size
    GPUSummary get_summary() {
        if (cpu_intervals_.size() >= GPU_THRESHOLD) {
            return compute_summary_gpu();
        } else {
            return compute_summary_cpu();
        }
    }
    
    // GPU-accelerated best-fit search
    std::optional<std::pair<int, int>> find_best_fit_gpu(int length, bool prefer_early = true) {
        sync_to_gpu();
        
        int n = gpu_intervals_.size();
        if (n == 0) return std::nullopt;
        
        // Initialize result storage
        thrust::device_vector<int> best_idx(1, INT_MAX);
        thrust::device_vector<int> best_waste(1, INT_MAX);
        
        int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        parallel_best_fit_kernel<<<num_blocks, BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(gpu_intervals_.data()),
            n,
            length,
            prefer_early,
            thrust::raw_pointer_cast(best_idx.data()),
            thrust::raw_pointer_cast(best_waste.data())
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        int idx = best_idx[0];
        if (idx == INT_MAX) return std::nullopt;
        
        GPUInterval result;
        CUDA_CHECK(cudaMemcpy(&result,
                             thrust::raw_pointer_cast(gpu_intervals_.data()) + idx,
                             sizeof(GPUInterval),
                             cudaMemcpyDeviceToHost));
        
        return std::make_pair(result.start, result.end);
    }
    
    // Performance statistics
    struct PerformanceStats {
        int total_operations;
        int gpu_operations;
        int cpu_operations;
        double gpu_utilization;
        size_t gpu_memory_used;
    };
    
    PerformanceStats get_performance_stats() const {
        size_t gpu_memory = (gpu_intervals_.size() * sizeof(GPUInterval)) +
                           (gpu_lengths_.size() * sizeof(int)) +
                           (gpu_gaps_.size() * sizeof(int));
        
        return {
            operation_count_,
            gpu_operations_,
            cpu_operations_,
            static_cast<double>(gpu_operations_) / std::max(1, operation_count_),
            gpu_memory
        };
    }
    
    // Debug info
    void print_info() const {
        std::cout << "GPU Boundary Summary Manager\n";
        std::cout << "  CPU intervals: " << cpu_intervals_.size() << "\n";
        std::cout << "  GPU intervals: " << gpu_intervals_.size() << "\n";
        std::cout << "  GPU threshold: " << GPU_THRESHOLD << "\n";
        
        auto stats = get_performance_stats();
        std::cout << "  Total operations: " << stats.total_operations << "\n";
        std::cout << "  GPU operations: " << stats.gpu_operations << "\n";
        std::cout << "  CPU operations: " << stats.cpu_operations << "\n";
        std::cout << "  GPU utilization: " << (stats.gpu_utilization * 100) << "%\n";
        std::cout << "  GPU memory: " << (stats.gpu_memory_used / 1024.0) << " KB\n";
    }

private:
    void sync_to_gpu() {
        if (!gpu_dirty_) return;
        
        // Copy intervals from CPU map to GPU vector
        thrust::host_vector<GPUInterval> host_intervals;
        host_intervals.reserve(cpu_intervals_.size());
        
        for (const auto& [start, end] : cpu_intervals_) {
            host_intervals.push_back({start, end});
        }
        
        gpu_intervals_ = host_intervals;
        gpu_dirty_ = false;
    }
    
    void update_managed_bounds(int start, int end) {
        if (managed_start_ == -1) {
            managed_start_ = start;
            managed_end_ = end;
        } else {
            managed_start_ = std::min(managed_start_, start);
            managed_end_ = std::max(managed_end_, end);
        }
    }
};