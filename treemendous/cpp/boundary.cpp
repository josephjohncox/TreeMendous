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

class IntervalManager {
public:
    IntervalManager() : total_available_length(0) {}

    void release_interval(Coordinate start, Coordinate end) {
        treemendous::validate_span(start, end);

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

        const Measure merged_length = treemendous::checked_length(merged_start, merged_end);
        const Measure retained = total_available_length - removed;
        const Measure prospective_total = treemendous::checked_add(retained, merged_length);

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

    void reserve_interval(Coordinate start, Coordinate end) {
        treemendous::validate_span(start, end);

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
                    removed, treemendous::checked_length(overlap_start, overlap_end));
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
            auto [position, inserted] = intervals.emplace(end, right_end);
            if (!inserted) throw std::logic_error("canonical interval split collision");
            right = position;
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

    std::vector<std::pair<Coordinate, Coordinate>> get_intervals() const {
        return {intervals.begin(), intervals.end()};
    }

    void print_intervals() const {
        for (const auto& [start, end] : intervals) std::cout << "[" << start << ", " << end << ")\n";
    }

private:
    std::map<Coordinate, Coordinate> intervals;
    Measure total_available_length;
};
