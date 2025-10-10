// Metal Performance Shaders Header for Boundary Summary
// C++ interface for the Objective-C++ implementation

#pragma once

#include <vector>
#include <optional>
#include <utility>
#include <climits>
#include <map>
#include <string>

// ============================================================================
// DATA STRUCTURES
// ============================================================================

struct MetalInterval {
    int start;
    int end;
    
    int length() const { return end - start; }
};

struct MetalSummary {
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
    
    void update_metrics();
};

// ============================================================================
// METAL BOUNDARY SUMMARY MANAGER
// ============================================================================

class MetalBoundarySummaryManager {
public:
    struct PerformanceStats {
        size_t total_operations = 0;
        size_t gpu_operations = 0;
        double gpu_utilization = 0.0;
        size_t gpu_memory_used = 0;
    };

    MetalBoundarySummaryManager();
    ~MetalBoundarySummaryManager();

    // Prevent copying (due to Metal device ownership)
    MetalBoundarySummaryManager(const MetalBoundarySummaryManager&) = delete;
    MetalBoundarySummaryManager& operator=(const MetalBoundarySummaryManager&) = delete;

    // Core interval operations
    void release_interval(int start, int end);
    void reserve_interval(int start, int end);

    // Summary operations
    MetalSummary get_summary();
    MetalSummary compute_summary_gpu();
    MetalSummary compute_summary_cpu();

    // Advanced queries
    std::optional<std::pair<int, int>> find_best_fit_gpu(int length, bool prefer_early = true);
    
    // Interval access
    std::vector<std::pair<int, int>> get_intervals() const;

    // Performance and debugging
    PerformanceStats get_performance_stats() const;
    void print_info() const;

private:
    class Impl;  // Forward declaration for pimpl
    Impl* pImpl;  // Pointer to implementation
};

// ============================================================================
// MODULE-LEVEL UTILITIES
// ============================================================================

// Get Metal device information as key-value pairs
std::map<std::string, std::string> get_metal_device_info();

