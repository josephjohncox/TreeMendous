// Canonical portable CPU boundary interval manager.
#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <utility>
#include <vector>

#include "interval_types.h"

using treemendous::Coordinate;
using treemendous::Measure;

struct MutationDelta {
    std::vector<std::pair<Coordinate, Coordinate>> changed;
    Measure changed_length = 0;
    bool fully_covered = false;
};

class IntervalManager {
public:
    IntervalManager() : total_available_length(0) {}

    void set_managed_domain(
        const std::vector<std::pair<Coordinate, Coordinate>>& spans
    ) {
        if (!intervals.empty()) {
            throw std::logic_error("managed domain must be configured before mutation");
        }
        std::vector<std::pair<Coordinate, Coordinate>> ordered = spans;
        for (const auto& [start, end] : ordered) {
            treemendous::validate_span(start, end);
        }
        std::sort(ordered.begin(), ordered.end());
        std::vector<std::pair<Coordinate, Coordinate>> normalized;
        normalized.reserve(ordered.size());
        for (const auto& [start, end] : ordered) {
            if (normalized.empty() || normalized.back().second < start) {
                normalized.emplace_back(start, end);
                continue;
            }
            normalized.back().second = std::max(
                normalized.back().second, end);
        }
        if (normalized.empty()) {
            throw std::invalid_argument("managed domain must contain at least one span");
        }
        managed_domain.swap(normalized);
    }

    void release_interval(Coordinate start, Coordinate end) {
        mutate_release(start, end);
    }

    MutationDelta preview_release_delta(
        Coordinate start, Coordinate end
    ) const {
        treemendous::validate_span(start, end);
        validate_managed_span(start, end);
        MutationDelta result;
        Coordinate cursor = start;
        auto it = intervals.lower_bound(start);
        if (it != intervals.begin()) {
            auto previous = std::prev(it);
            if (previous->second > start) it = previous;
        }
        for (; it != intervals.end() && it->first < end; ++it) {
            if (it->second <= cursor) continue;
            if (it->first > cursor) {
                append_changed(result, cursor, std::min(it->first, end));
            }
            cursor = std::max(cursor, std::min(it->second, end));
            if (cursor >= end) break;
        }
        if (cursor < end) append_changed(result, cursor, end);
        result.fully_covered = result.changed_length == 0;
        return result;
    }

    void reserve_interval(Coordinate start, Coordinate end) {
        mutate_reserve(start, end);
    }

    MutationDelta preview_reserve_delta(
        Coordinate start, Coordinate end, bool require_covered
    ) const {
        treemendous::validate_span(start, end);
        validate_managed_span(start, end);
        MutationDelta result;
        bool coverage_contiguous = true;
        Coordinate coverage_cursor = start;
        auto it = intervals.lower_bound(start);
        if (it != intervals.begin()) {
            auto previous = std::prev(it);
            if (previous->second > start) it = previous;
        }
        for (; it != intervals.end() && it->first < end; ++it) {
            const Coordinate overlap_start = std::max(start, it->first);
            const Coordinate overlap_end = std::min(end, it->second);
            if (overlap_start >= overlap_end) continue;
            append_changed(result, overlap_start, overlap_end);
            if (overlap_start > coverage_cursor) coverage_contiguous = false;
            coverage_cursor = std::max(coverage_cursor, overlap_end);
        }
        result.fully_covered = coverage_contiguous && coverage_cursor >= end;
        if (require_covered && !result.fully_covered) {
            result.changed.clear();
            result.changed_length = 0;
        }
        return result;
    }

    std::pair<Coordinate, Coordinate> find_interval(Coordinate point, Measure length) const {
        treemendous::validate_length(length);
        auto it = intervals.upper_bound(point);
        if (it != intervals.begin()) {
            auto previous = std::prev(it);
            const Coordinate allocation_start = std::max(point, previous->first);
            if (allocation_start < previous->second &&
                treemendous::checked_length(allocation_start, previous->second) >= length) {
                return {allocation_start, treemendous::checked_end(allocation_start, length)};
            }
        }
        for (; it != intervals.end(); ++it) {
            const Coordinate allocation_start = std::max(point, it->first);
            if (allocation_start < it->second &&
                treemendous::checked_length(allocation_start, it->second) >= length) {
                return {allocation_start, treemendous::checked_end(allocation_start, length)};
            }
        }
        return {0, 0};
    }

    Measure get_total_available_length() const { return total_available_length; }

    std::size_t get_interval_count() const { return intervals.size(); }

    Measure get_largest_available_length() const {
        Measure largest = 0;
        for (const auto& [start, end] : intervals) {
            largest = std::max(
                largest, treemendous::checked_length(start, end));
        }
        return largest;
    }

    std::vector<std::pair<Coordinate, Coordinate>> get_intervals() const {
        return {intervals.begin(), intervals.end()};
    }

    std::vector<std::pair<Coordinate, Coordinate>> find_overlapping_intervals(
        Coordinate start, Coordinate end
    ) const {
        treemendous::validate_span(start, end);
        validate_managed_span(start, end);
        std::vector<std::pair<Coordinate, Coordinate>> result;
        auto it = intervals.lower_bound(start);
        if (it != intervals.begin()) {
            auto previous = std::prev(it);
            if (previous->second > start) it = previous;
        }
        for (; it != intervals.end() && it->first < end; ++it) {
            if (it->second > start) result.emplace_back(it->first, it->second);
        }
        return result;
    }

    void print_intervals() const {
        for (const auto& [start, end] : intervals) std::cout << "[" << start << ", " << end << ")\n";
    }

private:
    static void append_changed(
        MutationDelta& result, Coordinate start, Coordinate end
    ) {
        if (start >= end) return;
        result.changed.emplace_back(start, end);
        result.changed_length = treemendous::checked_add(
            result.changed_length, treemendous::checked_length(start, end));
    }

    void mutate_release(Coordinate start, Coordinate end) {
        treemendous::validate_span(start, end);
        validate_managed_span(start, end);

        auto first = intervals.lower_bound(start);
        if (first != intervals.begin()) {
            auto previous = std::prev(first);
            if (previous->second >= start) first = previous;
        }
        if (first != intervals.end() && first->first <= start && end <= first->second) {
            return;
        }

        Coordinate merged_start = start;
        Coordinate merged_end = end;
        Measure removed = 0;
        auto last = first;
        while (last != intervals.end() && last->first <= merged_end) {
            merged_start = std::min(merged_start, last->first);
            merged_end = std::max(merged_end, last->second);
            removed = treemendous::checked_add(
                removed, treemendous::checked_length(last->first, last->second));
            ++last;
        }

        const Measure merged_length = treemendous::checked_length(
            merged_start, merged_end
        );
        const Measure retained = total_available_length - removed;
        const Measure prospective_total = treemendous::checked_add(
            retained, merged_length
        );

        // Allocate a new node, when needed, before erasing committed nodes.
        // Integer assignment and map erasure are then the no-throw commit phase.
        auto replacement = intervals.find(merged_start);
        if (replacement == intervals.end()) {
            intervals.emplace(merged_start, merged_end);
            intervals.erase(first, last);
        } else {
            replacement->second = merged_end;
            intervals.erase(std::next(replacement), last);
        }
        total_available_length = prospective_total;
    }

    void mutate_reserve(Coordinate start, Coordinate end) {
        treemendous::validate_span(start, end);
        validate_managed_span(start, end);

        auto first = intervals.lower_bound(start);
        if (first != intervals.begin()) {
            auto previous = std::prev(first);
            if (previous->second > start) first = previous;
        }

        Measure removed = 0;
        bool retain_left = false;
        bool retain_right = false;
        Coordinate right_end = end;
        auto last = first;
        while (last != intervals.end() && last->first < end) {
            const Coordinate overlap_start = std::max(start, last->first);
            const Coordinate overlap_end = std::min(end, last->second);
            if (overlap_start < overlap_end) {
                removed = treemendous::checked_add(
                    removed,
                    treemendous::checked_length(overlap_start, overlap_end)
                );
                if (last->first < start) retain_left = true;
                if (last->second > end) {
                    retain_right = true;
                    right_end = last->second;
                }
            }
            ++last;
        }
        if (removed == 0) return;

        const Measure prospective_total = total_available_length - removed;

        // A right split needs one new node. Allocate it before changing or
        // erasing any committed node so allocation failure remains atomic.
        auto right = intervals.end();
        if (retain_right) {
            auto insertion = intervals.emplace(end, right_end);
            if (!insertion.second) {
                throw std::logic_error("canonical interval split collision");
            }
            right = insertion.first;
        }
        auto current = first;
        while (current != last) {
            if ((retain_left && current == first) || current == right) {
                ++current;
            } else {
                current = intervals.erase(current);
            }
        }
        if (retain_left) first->second = start;
        total_available_length = prospective_total;
    }

    void validate_managed_span(Coordinate start, Coordinate end) const {
        if (managed_domain.empty()) return;
        auto it = std::upper_bound(
            managed_domain.begin(),
            managed_domain.end(),
            start,
            [](Coordinate value, const auto& span) {
                return value < span.first;
            }
        );
        if (it == managed_domain.begin()) {
            throw std::invalid_argument("span must be contained in the managed domain");
        }
        --it;
        if (start < it->first || end > it->second) {
            throw std::invalid_argument("span must be contained in the managed domain");
        }
    }

    std::map<Coordinate, Coordinate> intervals;
    std::vector<std::pair<Coordinate, Coordinate>> managed_domain;
    Measure total_available_length;
};
