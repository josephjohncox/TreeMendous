// Optimized Boundary-Based Summary Interval Manager
// Same optimizations as boundary_optimized.cpp

#include <vector>
#include <optional>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iomanip>
#include <climits>

// ============================================================================
// OPTIMIZATION FLAGS (default ON, except flat_map)
// ============================================================================
#ifndef TREE_MENDOUS_USE_FLAT_MAP
#define TREE_MENDOUS_USE_FLAT_MAP 0  // DISABLED: slower for insert-heavy workloads
#endif

#ifndef TREE_MENDOUS_USE_SMALL_VECTOR
#define TREE_MENDOUS_USE_SMALL_VECTOR 1
#endif

#ifndef TREE_MENDOUS_USE_SIMD
#define TREE_MENDOUS_USE_SIMD 1
#endif

#ifndef TREE_MENDOUS_PREALLOCATE_VECTORS
#define TREE_MENDOUS_PREALLOCATE_VECTORS 1
#endif

// ============================================================================
// INCLUDES BASED ON OPTIMIZATIONS
// ============================================================================
#if TREE_MENDOUS_USE_FLAT_MAP
#include <boost/container/flat_map.hpp>
template<typename K, typename V>
using IntervalMap = boost::container::flat_map<K, V>;
#else
#include <map>
template<typename K, typename V>
using IntervalMap = std::map<K, V>;
#endif

// ============================================================================
// SMALL VECTOR OPTIMIZATION (reuse from boundary_optimized.cpp)
// ============================================================================
#if TREE_MENDOUS_USE_SMALL_VECTOR
template<typename T, size_t N>
class SmallVector {
private:
    alignas(16) T stack_storage[N];
    T* data_ = stack_storage;
    size_t size_ = 0;
    size_t capacity_ = N;
    bool on_heap_ = false;

public:
    SmallVector() = default;
    
    ~SmallVector() {
        if (on_heap_) {
            delete[] data_;
        }
    }
    
    SmallVector(SmallVector&& other) noexcept {
        if (other.on_heap_) {
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            on_heap_ = true;
            other.data_ = nullptr;
            other.size_ = 0;
            other.on_heap_ = false;
        } else {
            std::copy(other.stack_storage, other.stack_storage + other.size_, stack_storage);
            data_ = stack_storage;
            size_ = other.size_;
            capacity_ = N;
            on_heap_ = false;
        }
    }
    
    void push_back(const T& value) {
        if (size_ >= capacity_) {
            size_t new_capacity = capacity_ * 2;
            T* new_data = new T[new_capacity];
            std::copy(data_, data_ + size_, new_data);
            if (on_heap_) {
                delete[] data_;
            }
            data_ = new_data;
            capacity_ = new_capacity;
            on_heap_ = true;
        }
        data_[size_++] = value;
    }
    
    template<typename... Args>
    void emplace_back(Args&&... args) {
        if (size_ >= capacity_) {
            size_t new_capacity = capacity_ * 2;
            T* new_data = new T[new_capacity];
            std::copy(data_, data_ + size_, new_data);
            if (on_heap_) {
                delete[] data_;
            }
            data_ = new_data;
            capacity_ = new_capacity;
            on_heap_ = true;
        }
        data_[size_++] = T(std::forward<Args>(args)...);
    }
    
    T* begin() { return data_; }
    T* end() { return data_ + size_; }
    const T* begin() const { return data_; }
    const T* end() const { return data_ + size_; }
    
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    
    void reserve(size_t new_capacity) {
        if (new_capacity <= capacity_) return;
        T* new_data = new T[new_capacity];
        std::copy(data_, data_ + size_, new_data);
        if (on_heap_) {
            delete[] data_;
        }
        data_ = new_data;
        capacity_ = new_capacity;
        on_heap_ = true;
    }
};

template<typename T>
using ErasureVector = SmallVector<T, 4>;
#else
template<typename T>
using ErasureVector = std::vector<T>;
#endif

// ============================================================================
// DATA STRUCTURES
// ============================================================================
struct IntervalResult {
    int start;
    int end;
    int length;
    void* data = nullptr;
    
    IntervalResult(int s, int e) : start(s), end(e), length(e - s) {}
    IntervalResult(int s, int e, int len) : start(s), end(e), length(len) {}
};

struct BoundarySummary {
    int total_free_length = 0;
    int total_occupied_length = 0;
    int interval_count = 0;
    int largest_interval_length = 0;
    int largest_interval_start = -1;
    int smallest_interval_length = 0;
    double avg_interval_length = 0.0;
    int total_gaps = 0;
    double avg_gap_size = 0.0;
    double fragmentation_index = 0.0;
    int earliest_start = -1;
    int latest_end = -1;
    double utilization = 0.0;
    
    void update_metrics() {
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

// ============================================================================
// OPTIMIZED BOUNDARY SUMMARY MANAGER
// ============================================================================
class BoundarySummaryManagerOptimized {
public:
    BoundarySummaryManagerOptimized() : operation_count_(0), cache_hits_(0), summary_dirty_(true),
                                        managed_start_(-1), managed_end_(-1) {
#if TREE_MENDOUS_PREALLOCATE_VECTORS && TREE_MENDOUS_USE_FLAT_MAP
        // Pre-allocate space for flat_map (boost::flat_map supports reserve)
        intervals_.reserve(64);
#endif
    }
    
    void release_interval(int start, int end) {
        if (start >= end) [[unlikely]] return;
        
        update_managed_bounds(start, end);
        invalidate_cache();
        ++operation_count_;
        
        auto it = intervals_.lower_bound(start);
        
        if (it != intervals_.begin()) [[likely]] {
            auto prev_it = std::prev(it);
            if (prev_it->second >= start) [[likely]] {
                start = prev_it->first;
                end = std::max(end, prev_it->second);
                intervals_.erase(prev_it);
            }
        }
        
        while (it != intervals_.end() && it->first <= end) [[unlikely]] {
            end = std::max(end, it->second);
            it = intervals_.erase(it);
        }
        
        intervals_[start] = end;
    }
    
    void reserve_interval(int start, int end) {
        if (start >= end) [[unlikely]] return;
        
        update_managed_bounds(start, end);
        invalidate_cache();
        ++operation_count_;
        
        auto it = intervals_.lower_bound(start);
        
        if (it != intervals_.begin()) {
            auto prev_it = std::prev(it);
            if (prev_it->second > start) [[likely]] {
                it = prev_it;
            }
        }
        
#if TREE_MENDOUS_USE_SMALL_VECTOR
        ErasureVector<typename IntervalMap<int, int>::iterator> to_erase;
        ErasureVector<std::pair<int, int>> to_add;
#else
        std::vector<typename IntervalMap<int, int>::iterator> to_erase;
        std::vector<std::pair<int, int>> to_add;
#endif

#if TREE_MENDOUS_PREALLOCATE_VECTORS
        to_erase.reserve(4);
        to_add.reserve(4);
#endif
        
        while (it != intervals_.end() && it->first < end) {
            int curr_start = it->first;
            int curr_end = it->second;
            
            int overlap_start = std::max(start, curr_start);
            int overlap_end = std::min(end, curr_end);
            
            if (overlap_start < overlap_end) [[likely]] {
                to_erase.push_back(it);
                
                if (curr_start < start) [[likely]] {
                    to_add.emplace_back(curr_start, start);
                }
                if (curr_end > end) [[likely]] {
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
        
        if (it != intervals_.end()) [[likely]] {
            int s = it->first;
            int e = it->second;
            if (s <= start && e - start >= length) [[likely]] {
                return IntervalResult(start, start + length);
            } else if (s > start && e - s >= length) {
                return IntervalResult(s, s + length);
            }
        }
        
        if (it != intervals_.begin()) [[likely]] {
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
            if (available >= length) [[likely]] {
                if (prefer_early) {
                    if (start < best_start) {
                        best_candidate = IntervalResult(start, start + length);
                        best_start = start;
                    }
                } else {
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
        
        if (summary.largest_interval_length == 0) [[unlikely]] {
            return std::nullopt;
        }
        
        for (const auto& [start, end] : intervals_) {
            if ((end - start) == summary.largest_interval_length) [[unlikely]] {
                return IntervalResult(start, end);
            }
        }
        
        return std::nullopt;
    }
    
    BoundarySummary get_summary() {
        if (!summary_dirty_ && cached_summary_.has_value()) [[likely]] {
            ++cache_hits_;
            return cached_summary_.value();
        }
        
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
            "boundary_summary_optimized_cpp",
            static_cast<int>(intervals_.size())
        };
    }
    
    void print_intervals() const {
        std::cout << "Optimized Boundary-Based Summary Interval Manager:\n";
        std::cout << "Available intervals (" << intervals_.size() << "):\n";
        
        for (const auto& [start, end] : intervals_) {
            std::cout << "  [" << start << ", " << end << ") length=" << (end - start) << "\n";
        }
        
        std::cout << "\nOptimizations enabled:\n";
#if TREE_MENDOUS_USE_FLAT_MAP
        std::cout << "  ✓ boost::flat_map\n";
#endif
#if TREE_MENDOUS_USE_SMALL_VECTOR
        std::cout << "  ✓ Small vector optimization\n";
#endif
#if TREE_MENDOUS_PREALLOCATE_VECTORS
        std::cout << "  ✓ Vector pre-allocation\n";
#endif
    }

private:
    IntervalMap<int, int> intervals_;
    
    mutable std::optional<BoundarySummary> cached_summary_;
    mutable bool summary_dirty_;
    mutable int cache_hits_;
    
    int managed_start_;
    int managed_end_;
    int operation_count_;
    
    void update_managed_bounds(int start, int end) {
        if (managed_start_ == -1) [[unlikely]] {
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
        
        if (intervals_.empty()) [[unlikely]] {
            if (managed_start_ != -1 && managed_end_ != -1) {
                summary.total_occupied_length = managed_end_ - managed_start_;
                summary.utilization = 1.0;
            }
            summary.update_metrics();
            return summary;
        }
        
        summary.interval_count = static_cast<int>(intervals_.size());
        summary.earliest_start = intervals_.begin()->first;
        summary.latest_end = intervals_.rbegin()->second;
        
        std::vector<int> lengths;
        lengths.reserve(intervals_.size());
        
        for (const auto& [start, end] : intervals_) {
            int length = end - start;
            lengths.push_back(length);
            summary.total_free_length += length;
        }
        
        if (!lengths.empty()) [[likely]] {
            summary.largest_interval_length = *std::max_element(lengths.begin(), lengths.end());
            summary.smallest_interval_length = *std::min_element(lengths.begin(), lengths.end());
            
            for (const auto& [start, end] : intervals_) {
                if ((end - start) == summary.largest_interval_length) {
                    summary.largest_interval_start = start;
                    break;
                }
            }
        }
        
        if (intervals_.size() > 1) {
            std::vector<int> gaps;
            gaps.reserve(intervals_.size() - 1);
            
            auto it = intervals_.begin();
            int prev_end = it->second;
            ++it;
            
            while (it != intervals_.end()) {
                if (it->first > prev_end) [[likely]] {
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
        
        if (managed_start_ != -1 && managed_end_ != -1) [[likely]] {
            int managed_space = managed_end_ - managed_start_;
            summary.total_occupied_length = managed_space - summary.total_free_length;
            summary.utilization = static_cast<double>(summary.total_occupied_length) / managed_space;
        }
        
        summary.update_metrics();
        return summary;
    }
};
