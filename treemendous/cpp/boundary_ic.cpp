// IntervalManager class implementation using Boost Interval Containers
#include <boost/icl/interval_set.hpp>
#include <boost/icl/interval.hpp>
#include <iostream>
#include <vector>
#include <optional>

class ICIntervalManager {
public:
    ICIntervalManager() : total_available_length(0) {}

    void release_interval(int start, int end) {
        if (start >= end) return;
        auto interval = boost::icl::interval<int>::right_open(start, end);

        intervals.add(interval);

        total_available_length = intervals.size();
    }

    void reserve_interval(int start, int end) {
        if (start >= end) return;
        auto interval = boost::icl::interval<int>::right_open(start, end);

        intervals.subtract(interval);

        total_available_length = intervals.size();
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
        return std::nullopt;
    }

    int get_total_available_length() const {
        return total_available_length;
    }

    void print_intervals() const {
        std::cout << "Available intervals:\n";
        for (const auto& interval : intervals) {
            std::cout << "[" << interval.lower() << ", " << interval.upper() << ")\n";
        }
        std::cout << "Total available length: " << total_available_length << "\n";
    }

    std::vector<std::pair<int, int>> get_intervals() const {
        std::vector<std::pair<int, int>> result;
        for (const auto& interval : intervals) {
            result.emplace_back(interval.lower(), interval.upper());
        }
        return result;
    }

private:
    boost::icl::interval_set<int> intervals;
    int total_available_length;
};