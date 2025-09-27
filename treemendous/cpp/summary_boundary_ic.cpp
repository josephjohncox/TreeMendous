// Summary-Enhanced ICIntervalManager using Boost Interval Containers
#include <boost/icl/interval_set.hpp>
#include <boost/icl/interval.hpp>
#include <iostream>
#include <vector>
#include <optional>
#include <algorithm>
#include <cmath>

// Same TreeSummary structure as in summary_boundary.cpp
struct TreeSummary {
    int total_free_length = 0;
    int total_occupied_length = 0;
    int contiguous_count = 0;
    int largest_free_length = 0;
    int largest_free_start = -1;
    int earliest_free_start = -1;
    int latest_free_end = -1;
    double avg_free_length = 0.0;
    double free_density = 0.0;
    
    // Calculated metrics
    double utilization = 0.0;
    double fragmentation = 0.0;
    
    void update_metrics() {
        int total_space = total_free_length + total_occupied_length;
        utilization = total_space > 0 ? static_cast<double>(total_occupied_length) / total_space : 0.0;
        free_density = total_space > 0 ? static_cast<double>(total_free_length) / total_space : 0.0;
        avg_free_length = contiguous_count > 0 ? static_cast<double>(total_free_length) / contiguous_count : 0.0;
        fragmentation = total_free_length > 0 ? 
            1.0 - static_cast<double>(largest_free_length) / total_free_length : 0.0;
    }
};

class SummaryICIntervalManager {
public:
    SummaryICIntervalManager() : total_space_tracked(0) {}
    
    void release_interval(int start, int end) {
        if (start >= end) return;
        
        track_space_bounds(start, end);
        
        auto interval = boost::icl::interval<int>::right_open(start, end);
        intervals.add(interval);
        
        update_summary();
    }
    
    void reserve_interval(int start, int end) {
        if (start >= end) return;
        
        track_space_bounds(start, end);
        
        auto interval = boost::icl::interval<int>::right_open(start, end);
        intervals.subtract(interval);
        
        update_summary();
    }
    
    std::optional<std::pair<int, int>> find_interval(int point, int length) {
        auto it = intervals.find(point);
        if (it != intervals.end()) {
            int s = it->lower();
            int e = it->upper();
            if (e - point >= length) {
                return std::make_pair(point, point + length);
            }
        }
        
        // Search for first suitable interval starting at or after point
        for (const auto& interval : intervals) {
            int s = interval.lower();
            int e = interval.upper();
            int available = e - s;
            
            if (s >= point && available >= length) {
                return std::make_pair(s, s + length);
            } else if (s <= point && point < e && (e - point) >= length) {
                return std::make_pair(point, point + length);
            }
        }
        
        return std::nullopt;
    }
    
    // Enhanced query operations using summary statistics
    std::optional<std::pair<int, int>> find_best_fit(int length, bool prefer_early = true) {
        if (summary.largest_free_length < length) {
            return std::nullopt;  // Quick elimination using summary
        }
        
        std::optional<std::pair<int, int>> best_candidate;
        int best_fit_size = INT_MAX;
        int best_start = INT_MAX;
        
        for (const auto& interval : intervals) {
            int start = interval.lower();
            int end = interval.upper();
            int available = end - start;
            
            if (available >= length) {
                if (prefer_early) {
                    if (start < best_start) {
                        best_candidate = std::make_pair(start, start + length);
                        best_start = start;
                    }
                } else {
                    // Best fit: smallest interval that satisfies requirement
                    if (available < best_fit_size) {
                        best_candidate = std::make_pair(start, start + length);
                        best_fit_size = available;
                    }
                }
            }
        }
        
        return best_candidate;
    }
    
    std::optional<std::pair<int, int>> find_largest_available() {
        if (summary.largest_free_length == 0) {
            return std::nullopt;
        }
        
        // Find the interval with the largest size
        for (const auto& interval : intervals) {
            int start = interval.lower();
            int end = interval.upper();
            if ((end - start) == summary.largest_free_length) {
                return std::make_pair(start, end);
            }
        }
        
        return std::nullopt;
    }
    
    // Summary statistics access
    TreeSummary get_summary() const {
        return summary;
    }
    
    // Legacy compatibility methods
    int get_total_available_length() const {
        return summary.total_free_length;
    }
    
    void print_intervals() const {
        std::cout << "Available intervals (Boost ICL):\n";
        for (const auto& interval : intervals) {
            std::cout << "[" << interval.lower() << ", " << interval.upper() 
                     << ") length=" << (interval.upper() - interval.lower()) << "\n";
        }
        std::cout << "Total available length: " << summary.total_free_length << "\n";
        std::cout << "Summary Statistics:\n";
        std::cout << "  Free chunks: " << summary.contiguous_count << "\n";
        std::cout << "  Largest chunk: " << summary.largest_free_length << "\n";
        std::cout << "  Utilization: " << (summary.utilization * 100) << "%\n";
        std::cout << "  Fragmentation: " << (summary.fragmentation * 100) << "%\n";
    }
    
    std::vector<std::pair<int, int>> get_intervals() const {
        std::vector<std::pair<int, int>> result;
        for (const auto& interval : intervals) {
            result.emplace_back(interval.lower(), interval.upper());
        }
        return result;
    }
    
    // Extended statistics interface
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
    };
    
    AvailabilityStats get_availability_stats() const {
        return {
            summary.total_free_length,
            summary.total_occupied_length,
            summary.total_free_length + summary.total_occupied_length,
            summary.contiguous_count,
            summary.largest_free_length,
            summary.avg_free_length,
            summary.utilization,
            summary.fragmentation,
            summary.free_density,
            {summary.earliest_free_start, summary.latest_free_end}
        };
    }

private:
    boost::icl::interval_set<int> intervals;
    TreeSummary summary;
    int total_space_tracked;
    
    void track_space_bounds(int start, int end) {
        // Track the bounds of space we're managing
        if (total_space_tracked == 0) {
            total_space_tracked = end - start;
        }
    }
    
    void update_summary() {
        summary = TreeSummary{};  // Reset
        
        if (intervals.empty()) {
            summary.total_occupied_length = total_space_tracked;
            summary.update_metrics();
            return;
        }
        
        // Calculate free space metrics using Boost ICL
        summary.contiguous_count = static_cast<int>(intervals.size());
        
        if (!intervals.empty()) {
            summary.earliest_free_start = intervals.begin()->lower();
            summary.latest_free_end = intervals.rbegin()->upper();
        }
        
        int total_free = 0;
        int max_length = 0;
        int max_start = -1;
        
        for (const auto& interval : intervals) {
            int start = interval.lower();
            int end = interval.upper();
            int length = end - start;
            
            total_free += length;
            
            if (length > max_length) {
                max_length = length;
                max_start = start;
            }
        }
        
        summary.total_free_length = total_free;
        summary.largest_free_length = max_length;
        summary.largest_free_start = max_start;
        
        // Estimate occupied space
        if (total_space_tracked > 0) {
            summary.total_occupied_length = std::max(0, total_space_tracked - total_free);
        } else {
            summary.total_occupied_length = estimate_occupied_space();
        }
        
        summary.update_metrics();
    }
    
    int estimate_occupied_space() const {
        if (intervals.size() < 2) return 0;
        
        // Use Boost ICL's gap calculation
        int occupied = 0;
        auto it = intervals.begin();
        int prev_end = it->upper();
        ++it;
        
        while (it != intervals.end()) {
            int current_start = it->lower();
            if (current_start > prev_end) {
                occupied += current_start - prev_end;  // Gap = occupied space
            }
            prev_end = it->upper();
            ++it;
        }
        
        return occupied;
    }
};

// Alias for backward compatibility
using ICIntervalManager = SummaryICIntervalManager;
