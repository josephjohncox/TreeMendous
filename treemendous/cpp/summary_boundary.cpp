// Summary-Enhanced IntervalManager with comprehensive aggregate statistics
#include <map>
#include <vector>
#include <optional>
#include <iostream>
#include <algorithm>
#include <cmath>

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

class SummaryIntervalManager {
public:
    SummaryIntervalManager() : total_space_tracked(0) {}
    
    void release_interval(int start, int end) {
        if (start >= end) return;
        
        track_space_bounds(start, end);
        
        auto it = intervals.lower_bound(start);
        
        // Merge with previous interval if overlapping or adjacent
        if (it != intervals.begin()) {
            auto prev_it = std::prev(it);
            if (prev_it->second >= start) {
                start = prev_it->first;
                end = std::max(end, prev_it->second);
                intervals.erase(prev_it);
            }
        }
        
        // Merge with overlapping intervals
        while (it != intervals.end() && it->first <= end) {
            end = std::max(end, it->second);
            it = intervals.erase(it);
        }
        
        intervals[start] = end;
        update_summary();
    }
    
    void reserve_interval(int start, int end) {
        if (start >= end) return;
        
        track_space_bounds(start, end);
        
        auto it = intervals.lower_bound(start);
        
        if (it != intervals.begin()) {
            auto prev_it = std::prev(it);
            if (prev_it->second > start) {
                it = prev_it;
            }
        }
        
        std::vector<std::map<int, int>::iterator> to_erase;
        std::vector<std::pair<int, int>> to_add;
        
        while (it != intervals.end() && it->first < end) {
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
            intervals.erase(eit);
        }
        for (const auto& interval : to_add) {
            intervals[interval.first] = interval.second;
        }
        
        update_summary();
    }
    
    std::optional<std::pair<int, int>> find_interval(int point, int length) {
        auto it = intervals.lower_bound(point);
        
        if (it != intervals.end()) {
            int s = it->first;
            int e = it->second;
            if (s <= point && e - point >= length) {
                return std::make_pair(point, point + length);
            } else if (s > point && e - s >= length) {
                return std::make_pair(s, s + length);
            }
        }
        
        if (it != intervals.begin()) {
            --it;
            int s = it->first;
            int e = it->second;
            if (s <= point && e - point >= length) {
                return std::make_pair(point, point + length);
            } else if (point < s && e - s >= length) {
                return std::make_pair(s, s + length);
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
        
        for (const auto& [start, end] : intervals) {
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
        for (const auto& [start, end] : intervals) {
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
        std::cout << "Available intervals:\n";
        for (const auto& [s, e] : intervals) {
            std::cout << "[" << s << ", " << e << ") length=" << (e-s) << "\n";
        }
        std::cout << "Total available length: " << summary.total_free_length << "\n";
        std::cout << "Summary Statistics:\n";
        std::cout << "  Free chunks: " << summary.contiguous_count << "\n";
        std::cout << "  Largest chunk: " << summary.largest_free_length << "\n";
        std::cout << "  Utilization: " << (summary.utilization * 100) << "%\n";
        std::cout << "  Fragmentation: " << (summary.fragmentation * 100) << "%\n";
    }
    
    std::vector<std::pair<int, int>> get_intervals() const {
        return std::vector<std::pair<int, int>>(intervals.begin(), intervals.end());
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
    std::map<int, int> intervals;
    TreeSummary summary;
    int total_space_tracked;
    
    void track_space_bounds(int start, int end) {
        // Track the bounds of space we're managing
        if (total_space_tracked == 0) {
            total_space_tracked = end - start;
        } else {
            // Expand tracking bounds if necessary
            // This is a simplified approach - in practice you might want more sophisticated tracking
        }
    }
    
    void update_summary() {
        summary = TreeSummary{};  // Reset
        
        if (intervals.empty()) {
            summary.total_occupied_length = total_space_tracked;
            summary.update_metrics();
            return;
        }
        
        // Calculate free space metrics
        summary.contiguous_count = static_cast<int>(intervals.size());
        summary.earliest_free_start = intervals.begin()->first;
        summary.latest_free_end = intervals.rbegin()->second;
        
        int total_free = 0;
        int max_length = 0;
        int max_start = -1;
        
        for (const auto& [start, end] : intervals) {
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
        
        // Estimate occupied space (simplified approach)
        if (total_space_tracked > 0) {
            summary.total_occupied_length = std::max(0, total_space_tracked - total_free);
        } else {
            // Estimate based on gap analysis between intervals
            summary.total_occupied_length = estimate_occupied_space();
        }
        
        summary.update_metrics();
    }
    
    int estimate_occupied_space() const {
        if (intervals.size() < 2) return 0;
        
        int occupied = 0;
        auto it = intervals.begin();
        auto prev_end = it->second;
        ++it;
        
        while (it != intervals.end()) {
            if (it->first > prev_end) {
                occupied += it->first - prev_end;  // Gap = occupied space
            }
            prev_end = it->second;
            ++it;
        }
        
        return occupied;
    }
};

// Alias for backward compatibility
using IntervalManager = SummaryIntervalManager;
