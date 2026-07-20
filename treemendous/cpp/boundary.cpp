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

        // Work against a complete prospective state. Any checked length or
        // aggregate failure must leave both the map and cached total unchanged.
        auto prospective = intervals;
        auto it = prospective.lower_bound(start);
        if (it != prospective.begin()) {
            auto previous = std::prev(it);
            if (previous->second >= start) {
                start = previous->first;
                end = std::max(end, previous->second);
                prospective.erase(previous);
            }
        }
        it = prospective.lower_bound(start);
        while (it != prospective.end() && it->first <= end) {
            end = std::max(end, it->second);
            it = prospective.erase(it);
        }
        prospective[start] = end;

        const Measure prospective_total = checked_total(prospective);
        intervals.swap(prospective);
        total_available_length = prospective_total;
    }

    void reserve_interval(Coordinate start, Coordinate end) {
        treemendous::validate_span(start, end);

        auto prospective = intervals;
        auto it = prospective.lower_bound(start);
        if (it != prospective.begin()) {
            auto previous = std::prev(it);
            if (previous->second > start) it = previous;
        }
        std::vector<Coordinate> erase;
        std::vector<std::pair<Coordinate, Coordinate>> add;
        while (it != prospective.end() && it->first < end) {
            const Coordinate current_start = it->first;
            const Coordinate current_end = it->second;
            if (std::max(start, current_start) < std::min(end, current_end)) {
                erase.push_back(current_start);
                if (current_start < start) add.emplace_back(current_start, start);
                if (current_end > end) add.emplace_back(end, current_end);
            }
            ++it;
        }
        for (const Coordinate key : erase) prospective.erase(key);
        for (const auto& [part_start, part_end] : add) {
            prospective[part_start] = part_end;
        }

        const Measure prospective_total = checked_total(prospective);
        intervals.swap(prospective);
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
    static Measure checked_total(const std::map<Coordinate, Coordinate>& values) {
        Measure total = 0;
        for (const auto& [start, end] : values) {
            total = treemendous::checked_add(total, treemendous::checked_length(start, end));
        }
        return total;
    }

    std::map<Coordinate, Coordinate> intervals;
    Measure total_available_length;
};
