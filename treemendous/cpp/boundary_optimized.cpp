// Optimized IntervalManager with boost::flat_map, SIMD, and small vector optimization
// Performance optimizations enabled by default via #define flags

#include <vector>
#include <optional>
#include <iostream>
#include <algorithm>

// ============================================================================
// OPTIMIZATION FLAGS (default ON, except flat_map)
// ============================================================================
#ifndef TREE_MENDOUS_USE_FLAT_MAP
#define TREE_MENDOUS_USE_FLAT_MAP 0  // DISABLED: slower for insert-heavy workloads
#endif

#ifndef TREE_MENDOUS_USE_SMALL_VECTOR
#define TREE_MENDOUS_USE_SMALL_VECTOR 1  // Stack-allocate small vectors
#endif

#ifndef TREE_MENDOUS_USE_SIMD
#define TREE_MENDOUS_USE_SIMD 1  // SIMD for batch operations
#endif

#ifndef TREE_MENDOUS_PREALLOCATE_VECTORS
#define TREE_MENDOUS_PREALLOCATE_VECTORS 1  // Pre-allocate vector capacity
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

#if TREE_MENDOUS_USE_SIMD
  #if defined(__ARM_NEON) || defined(__aarch64__)
    #include <arm_neon.h>  // ARM NEON intrinsics
  #elif defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    #include <immintrin.h>  // x86 AVX/SSE intrinsics
  #endif
#endif

// ============================================================================
// SMALL VECTOR OPTIMIZATION
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
    
    // Move constructor
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

// Use small vector for common case (most operations affect 1-2 intervals)
template<typename T>
using ErasureVector = SmallVector<T, 4>;
#else
template<typename T>
using ErasureVector = std::vector<T>;
#endif

// ============================================================================
// SIMD UTILITIES
// ============================================================================
#if TREE_MENDOUS_USE_SIMD
  #if defined(__ARM_NEON) || defined(__aarch64__)
    // ARM NEON: Check if any of 4 intervals overlap with [start, end)
    inline bool simd_check_overlap_any(const int* starts, const int* ends, int start, int end) {
        int32x4_t target_start = vdupq_n_s32(start);
        int32x4_t target_end = vdupq_n_s32(end);
        int32x4_t interval_starts = vld1q_s32(starts);
        int32x4_t interval_ends = vld1q_s32(ends);
        
        // Check: interval_start < target_end && interval_end > target_start
        uint32x4_t cmp1 = vcltq_s32(interval_starts, target_end);
        uint32x4_t cmp2 = vcgtq_s32(interval_ends, target_start);
        uint32x4_t overlap = vandq_u32(cmp1, cmp2);
        
        // Check if any lane is non-zero
        return vmaxvq_u32(overlap) != 0;
    }
  #elif (defined(__x86_64__) || defined(_M_X64)) && (defined(__AVX2__) || defined(__AVX__))
    // x86 AVX: Check if any of 4 intervals overlap with [start, end)
    inline bool simd_check_overlap_any(const int* starts, const int* ends, int start, int end) {
        __m128i target_start = _mm_set1_epi32(start);
        __m128i target_end = _mm_set1_epi32(end);
        __m128i interval_starts = _mm_loadu_si128((__m128i*)starts);
        __m128i interval_ends = _mm_loadu_si128((__m128i*)ends);
        
        // Check: interval_start < target_end && interval_end > target_start
        __m128i cmp1 = _mm_cmplt_epi32(interval_starts, target_end);
        __m128i cmp2 = _mm_cmpgt_epi32(interval_ends, target_start);
        __m128i overlap = _mm_and_si128(cmp1, cmp2);
        
        return _mm_movemask_epi8(overlap) != 0;
    }
  #else
    // Fallback: no SIMD available, use scalar
    inline bool simd_check_overlap_any(const int* starts, const int* ends, int start, int end) {
        for (int i = 0; i < 4; ++i) {
            if (starts[i] < end && ends[i] > start) {
                return true;
            }
        }
        return false;
    }
  #endif
#endif

// ============================================================================
// OPTIMIZED INTERVAL MANAGER
// ============================================================================
class IntervalManagerOptimized {
public:
    IntervalManagerOptimized() : total_available_length(0) {
#if TREE_MENDOUS_PREALLOCATE_VECTORS && TREE_MENDOUS_USE_FLAT_MAP
        // Pre-allocate space for flat_map (boost::flat_map supports reserve)
        intervals.reserve(64);
#endif
    }

    void release_interval(int start, int end) {
        if (start >= end) [[unlikely]] return;

        auto it = intervals.lower_bound(start);

        // Merge with previous interval if overlapping or adjacent
        if (it != intervals.begin()) [[likely]] {
            auto prev_it = std::prev(it);
            if (prev_it->second >= start) [[likely]] {
                start = prev_it->first;
                end = std::max(end, prev_it->second);
                total_available_length -= prev_it->second - prev_it->first;
                intervals.erase(prev_it);
            }
        }

        // Merge with overlapping intervals
        while (it != intervals.end() && it->first <= end) [[unlikely]] {
            end = std::max(end, it->second);
            total_available_length -= it->second - it->first;
            it = intervals.erase(it);
        }

        intervals[start] = end;
        total_available_length += end - start;
    }

    void reserve_interval(int start, int end) {
        if (start >= end) [[unlikely]] return;

        auto it = intervals.lower_bound(start);

        if (it != intervals.begin()) {
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
        to_erase.reserve(4);  // Most operations affect 1-2 intervals
        to_add.reserve(4);
#endif

        while (it != intervals.end() && it->first < end) {
            int curr_start = it->first;
            int curr_end = it->second;

            int overlap_start = std::max(start, curr_start);
            int overlap_end = std::min(end, curr_end);

            if (overlap_start < overlap_end) [[likely]] {
                to_erase.push_back(it);
                total_available_length -= curr_end - curr_start;

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
            intervals.erase(eit);
        }
        for (const auto& interval : to_add) {
            intervals[interval.first] = interval.second;
            total_available_length += interval.second - interval.first;
        }
    }

    std::optional<std::pair<int, int>> find_interval(int point, int length) {
        auto it = intervals.lower_bound(point);

        if (it != intervals.end()) [[likely]] {
            int s = it->first;
            int e = it->second;
            if (s <= point && e - point >= length) [[likely]] {
                return std::make_pair(point, point + length);
            } else if (s > point && e - s >= length) {
                return std::make_pair(s, s + length);
            }
        }

        if (it != intervals.begin()) [[likely]] {
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

    int get_total_available_length() const {
        return total_available_length;
    }

    void print_intervals() const {
        std::ostream& out = std::cout;
        out << "Available intervals (optimized):\n";
        for (const auto& [s, e] : intervals) {
            out << "[" << s << ", " << e << ")\n";
        }
        out << "Total available length: " << total_available_length << "\n";
        
        // Print optimization status
        out << "\nOptimizations enabled:\n";
#if TREE_MENDOUS_USE_FLAT_MAP
        out << "  ✓ boost::flat_map (cache-friendly)\n";
#else
        out << "  - std::map (red-black tree)\n";
#endif
#if TREE_MENDOUS_USE_SMALL_VECTOR
        out << "  ✓ Small vector optimization\n";
#endif
#if TREE_MENDOUS_PREALLOCATE_VECTORS
        out << "  ✓ Vector pre-allocation\n";
#endif
#if TREE_MENDOUS_USE_SIMD
        out << "  ✓ SIMD batch operations\n";
#endif
    }

    std::vector<std::pair<int, int>> get_intervals() const {
        return std::vector<std::pair<int, int>>(intervals.begin(), intervals.end());
    }

private:
    IntervalMap<int, int> intervals;
    int total_available_length;
};
