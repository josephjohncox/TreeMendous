// Optimized Boundary-Based Summary Interval Manager
// Combines std::map efficiency with comprehensive summary statistics
// Provides O(log n) operations with O(1) cached summary queries

#include <map>
#include <vector>
#include <optional>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iomanip>
#include <climits>
#include <random>
#include <chrono>

struct IntervalResult {
    int start;
    int end;
    int length;
    void* data = nullptr;  // For compatibility, though we don't use it
    
    IntervalResult(int s, int e) : start(s), end(e), length(e - s) {}
    IntervalResult(int s, int e, int len) : start(s), end(e), length(len) {}
};

struct BoundarySummary {
    // Core metrics
    int total_free_length = 0;
    int total_occupied_length = 0;
    int interval_count = 0;
    
    // Efficiency metrics
    int largest_interval_length = 0;
    int largest_interval_start = -1;
    int smallest_interval_length = 0;
    double avg_interval_length = 0.0;
    
    // Space distribution
    int total_gaps = 0;
    double avg_gap_size = 0.0;
    double fragmentation_index = 0.0;
    
    // Bounds
    int earliest_start = -1;
    int latest_end = -1;
    
    // Utilization
    double utilization = 0.0;
    
    void update_metrics() {
        // Update calculated fields
        if (interval_count > 0) {
            avg_interval_length = static_cast<double>(total_free_length) / interval_count;
        } else {
            avg_interval_length = 0.0;
        }
        
        if (total_free_length > 0) {
            fragmentation_index = 1.0 - static_cast<double>(largest_interval_length) / total_free_length;
        } else {
            fragmentation_index = 0.0;
        }
    }
};

class BoundarySummaryManager {
public:
    BoundarySummaryManager() : operation_count_(0), cache_hits_(0), summary_dirty_(true),
                               managed_start_(-1), managed_end_(-1) {}
    
    void release_interval(int start, int end) {
        if (start >= end) return;
        
        update_managed_bounds(start, end);
        invalidate_cache();
        ++operation_count_;
        
        auto it = intervals_.lower_bound(start);
        
        // Merge with previous interval if overlapping or adjacent
        if (it != intervals_.begin()) {
            auto prev_it = std::prev(it);
            if (prev_it->second >= start) {
                start = prev_it->first;
                end = std::max(end, prev_it->second);
                intervals_.erase(prev_it);
            }
        }
        
        // Merge with overlapping intervals
        while (it != intervals_.end() && it->first <= end) {
            end = std::max(end, it->second);
            it = intervals_.erase(it);
        }
        
        intervals_[start] = end;
    }
    
    void reserve_interval(int start, int end) {
        if (start >= end) return;
        
        update_managed_bounds(start, end);
        invalidate_cache();
        ++operation_count_;
        
        auto it = intervals_.lower_bound(start);
        
        if (it != intervals_.begin()) {
            auto prev_it = std::prev(it);
            if (prev_it->second > start) {
                it = prev_it;
            }
        }
        
        std::vector<std::map<int, int>::iterator> to_erase;
        std::vector<std::pair<int, int>> to_add;
        
        while (it != intervals_.end() && it->first < end) {
            int curr_start = it->first;
            int curr_end = it->second;
            
            int overlap_start = std::max(start, curr_start);
            int overlap_end = std::min(end, curr_end);
            
            if (overlap_start < overlap_end) {
                to_erase.push_back(it);
                
                if (curr_start < start) {
                    to_add.emplace_back(curr_start, start);
                }
                if (curr_end > end) {
                    to_add.emplace_back(end, curr_end);
                }
            }
            ++it;
        }
        
        for (auto& eit : to_erase) {
            intervals_.erase(eit);
        }
        for (const auto& interval : to_add) {
            intervals_[interval.first] = interval.second;
        }
    }
    
    std::optional<IntervalResult> find_interval(int start, int length) {
        auto it = intervals_.lower_bound(start);
        
        if (it != intervals_.end()) {
            int s = it->first;
            int e = it->second;
            if (s <= start && e - start >= length) {
                return IntervalResult(start, start + length);
            } else if (s > start && e - s >= length) {
                return IntervalResult(s, s + length);
            }
        }
        
        if (it != intervals_.begin()) {
            --it;
            int s = it->first;
            int e = it->second;
            if (s <= start && e - start >= length) {
                return IntervalResult(start, start + length);
            } else if (start < s && e - s >= length) {
                return IntervalResult(s, s + length);
            }
        }
        
        return std::nullopt;
    }
    
    std::optional<IntervalResult> find_best_fit(int length, bool prefer_early = true) {
        std::optional<IntervalResult> best_candidate;
        int best_fit_size = INT_MAX;
        int best_start = INT_MAX;
        
        for (const auto& [start, end] : intervals_) {
            int available = end - start;
            if (available >= length) {
                if (prefer_early) {
                    if (start < best_start) {
                        best_candidate = IntervalResult(start, start + length);
                        best_start = start;
                    }
                } else {
                    // Best fit: smallest interval that satisfies requirement
                    if (available < best_fit_size) {
                        best_candidate = IntervalResult(start, start + length);
                        best_fit_size = available;
                    }
                }
            }
        }
        
        return best_candidate;
    }
    
    std::optional<IntervalResult> find_largest_available() {
        BoundarySummary summary = get_summary();
        
        if (summary.largest_interval_length == 0) {
            return std::nullopt;
        }
        
        // Find the interval with largest size
        for (const auto& [start, end] : intervals_) {
            if ((end - start) == summary.largest_interval_length) {
                return IntervalResult(start, end);
            }
        }
        
        return std::nullopt;
    }
    
    BoundarySummary get_summary() {
        if (!summary_dirty_ && cached_summary_.has_value()) {
            ++cache_hits_;
            return cached_summary_.value();
        }
        
        // Recompute summary
        cached_summary_ = compute_summary();
        summary_dirty_ = false;
        
        return cached_summary_.value();
    }
    
    struct AvailabilityStats {
        int total_free;
        int total_occupied;
        int total_space;
        int free_chunks;
        int largest_chunk;
        double avg_chunk_size;
        double utilization;
        double fragmentation;
        double free_density;
        std::pair<int, int> bounds;
        int gaps;
        double avg_gap_size;
    };
    
    AvailabilityStats get_availability_stats() {
        BoundarySummary summary = get_summary();
        
        return {
            summary.total_free_length,
            summary.total_occupied_length,
            summary.total_free_length + summary.total_occupied_length,
            summary.interval_count,
            summary.largest_interval_length,
            summary.avg_interval_length,
            summary.utilization,
            summary.fragmentation_index,
            1.0 - summary.utilization,
            {summary.earliest_start, summary.latest_end},
            summary.total_gaps,
            summary.avg_gap_size
        };
    }
    
    int get_total_available_length() {
        return get_summary().total_free_length;
    }
    
    std::vector<std::pair<int, int>> get_intervals() const {
        return std::vector<std::pair<int, int>>(intervals_.begin(), intervals_.end());
    }
    
    void print_intervals() const {
        std::cout << "Boundary-Based Summary Interval Manager (C++):\n";
        std::cout << "Available intervals (" << intervals_.size() << "):\n";
        
        for (const auto& [start, end] : intervals_) {
            std::cout << "  [" << start << ", " << end << ") length=" << (end - start) << "\n";
        }
        
        // Get summary (const_cast for demo purposes)
        BoundarySummary summary = const_cast<BoundarySummaryManager*>(this)->get_summary();
        
        std::cout << "\nSummary Statistics:\n";
        std::cout << "  Total free: " << summary.total_free_length << "\n";
        std::cout << "  Intervals: " << summary.interval_count << "\n";
        std::cout << "  Largest: " << summary.largest_interval_length << "\n";
        std::cout << "  Fragmentation: " << std::fixed << std::setprecision(2) << summary.fragmentation_index << "\n";
        std::cout << "  Utilization: " << std::fixed << std::setprecision(1) << (summary.utilization * 100) << "%\n";
    }
    
    struct PerformanceStats {
        int operation_count;
        int cache_hits;
        double cache_hit_rate;
        std::string implementation;
        int interval_count;
    };
    
    PerformanceStats get_performance_stats() const {
        return {
            operation_count_,
            cache_hits_,
            static_cast<double>(cache_hits_) / std::max(1, operation_count_),
            "boundary_summary_cpp",
            static_cast<int>(intervals_.size())
        };
    }

private:
    std::map<int, int> intervals_;
    
    // Summary caching
    mutable std::optional<BoundarySummary> cached_summary_;
    mutable bool summary_dirty_;
    mutable int cache_hits_;
    
    // Managed space tracking
    int managed_start_;
    int managed_end_;
    
    // Performance tracking
    int operation_count_;
    
    void update_managed_bounds(int start, int end) {
        if (managed_start_ == -1) {
            managed_start_ = start;
            managed_end_ = end;
        } else {
            managed_start_ = std::min(managed_start_, start);
            managed_end_ = std::max(managed_end_, end);
        }
    }
    
    void invalidate_cache() {
        summary_dirty_ = true;
    }
    
    BoundarySummary compute_summary() const {
        BoundarySummary summary;
        
        if (intervals_.empty()) {
            if (managed_start_ != -1 && managed_end_ != -1) {
                summary.total_occupied_length = managed_end_ - managed_start_;
                summary.utilization = 1.0;
            }
            summary.update_metrics();
            return summary;
        }
        
        // Compute basic metrics
        summary.interval_count = static_cast<int>(intervals_.size());
        summary.earliest_start = intervals_.begin()->first;
        summary.latest_end = intervals_.rbegin()->second;
        
        // Calculate lengths and find extremes
        std::vector<int> lengths;
        lengths.reserve(intervals_.size());
        
        for (const auto& [start, end] : intervals_) {
            int length = end - start;
            lengths.push_back(length);
            summary.total_free_length += length;
        }
        
        if (!lengths.empty()) {
            summary.largest_interval_length = *std::max_element(lengths.begin(), lengths.end());
            summary.smallest_interval_length = *std::min_element(lengths.begin(), lengths.end());
            
            // Find start of largest interval
            for (const auto& [start, end] : intervals_) {
                if ((end - start) == summary.largest_interval_length) {
                    summary.largest_interval_start = start;
                    break;
                }
            }
        }
        
        // Calculate gaps between intervals
        if (intervals_.size() > 1) {
            std::vector<int> gaps;
            auto it = intervals_.begin();
            int prev_end = it->second;
            ++it;
            
            while (it != intervals_.end()) {
                if (it->first > prev_end) {
                    gaps.push_back(it->first - prev_end);
                }
                prev_end = it->second;
                ++it;
            }
            
            summary.total_gaps = static_cast<int>(gaps.size());
            if (!gaps.empty()) {
                summary.avg_gap_size = static_cast<double>(std::accumulate(gaps.begin(), gaps.end(), 0)) / gaps.size();
            }
        }
        
        // Calculate utilization if managed space is tracked
        if (managed_start_ != -1 && managed_end_ != -1) {
            int managed_space = managed_end_ - managed_start_;
            summary.total_occupied_length = managed_space - summary.total_free_length;
            summary.utilization = static_cast<double>(summary.total_occupied_length) / managed_space;
        }
        
        summary.update_metrics();
        return summary;
    }
};

// Example usage and testing
#ifdef BOUNDARY_SUMMARY_STANDALONE_TEST
#include <chrono>
#include <random>

int main() {
    std::cout << "🏗️ C++ Boundary-Based Summary Manager Test\n";
    std::cout << "==========================================\n";
    
    BoundarySummaryManager manager;
    
    // Initialize with test space
    manager.release_interval(0, 10000);
    
    std::cout << "Initial state:\n";
    manager.print_intervals();
    
    // Apply test operations
    std::vector<std::tuple<std::string, int, int>> operations = {
        {"reserve", 1000, 1500},
        {"reserve", 3000, 3200},
        {"reserve", 5000, 5800},
        {"release", 2000, 2500},
        {"reserve", 7000, 7300},
        {"release", 6000, 6500}
    };
    
    std::cout << "\nApplying " << operations.size() << " operations...\n";
    
    for (const auto& [op, start, end] : operations) {
        if (op == "reserve") {
            manager.reserve_interval(start, end);
        } else {
            manager.release_interval(start, end);
        }
        std::cout << "  " << op << " [" << start << ", " << end << ")\n";
    }
    
    std::cout << "\nFinal state:\n";
    manager.print_intervals();
    
    // Test advanced queries
    std::cout << "\nAdvanced Queries:\n";
    
    auto best_fit = manager.find_best_fit(300);
    if (best_fit) {
        std::cout << "  Best fit (300 units): [" << best_fit->start << ", " << best_fit->end << ")\n";
    }
    
    auto largest = manager.find_largest_available();
    if (largest) {
        std::cout << "  Largest available: [" << largest->start << ", " << largest->end 
                  << "), size=" << largest->length << "\n";
    }
    
    auto perf = manager.get_performance_stats();
    std::cout << "  Performance: " << perf.operation_count << " ops, " 
              << std::fixed << std::setprecision(1) << (perf.cache_hit_rate * 100) << "% cache hit rate\n";
    
    // Performance benchmark
    std::cout << "\n⚡ Performance Benchmark\n";
    std::cout << "------------------------\n";
    
    BoundarySummaryManager perf_manager;
    perf_manager.release_interval(0, 100000);
    
    // Generate test operations
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> op_dist(0, 1);
    std::uniform_int_distribution<int> start_dist(0, 90000);
    std::uniform_int_distribution<int> length_dist(1, 1000);
    
    const int num_operations = 10000;
    std::vector<std::tuple<std::string, int, int>> perf_operations;
    
    for (int i = 0; i < num_operations; ++i) {
        std::string op = (op_dist(rng) == 0) ? "reserve" : "release";
        int start = start_dist(rng);
        int end = start + length_dist(rng);
        perf_operations.emplace_back(op, start, end);
    }
    
    // Benchmark operations
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (const auto& [op, start, end] : perf_operations) {
        if (op == "reserve") {
            perf_manager.reserve_interval(start, end);
        } else {
            perf_manager.release_interval(start, end);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Benchmark summary access
    auto summary_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        auto stats = perf_manager.get_availability_stats();
    }
    auto summary_end = std::chrono::high_resolution_clock::now();
    auto summary_duration = std::chrono::duration_cast<std::chrono::microseconds>(summary_end - summary_start);
    
    double ops_per_second = static_cast<double>(num_operations) / (duration.count() / 1000000.0);
    double avg_summary_time = static_cast<double>(summary_duration.count()) / 1000.0;
    
    std::cout << "Operations: " << num_operations << " in " << duration.count() << "µs\n";
    std::cout << "Ops/sec: " << std::fixed << std::setprecision(0) << ops_per_second << "\n";
    std::cout << "Summary queries: 1000 in " << summary_duration.count() << "µs (avg: " 
              << std::fixed << std::setprecision(1) << avg_summary_time << "µs)\n";
    
    auto final_perf = perf_manager.get_performance_stats();
    std::cout << "Cache hit rate: " << std::fixed << std::setprecision(1) 
              << (final_perf.cache_hit_rate * 100) << "%\n";
    
    std::cout << "\n✅ C++ Boundary Summary test complete!\n";
    
    return 0;
}
#endif
