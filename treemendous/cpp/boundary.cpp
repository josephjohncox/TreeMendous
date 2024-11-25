// IntervalManager class implementation
#include <map>
#include <vector>
#include <optional>
#include <iostream>

class IntervalManager {
public:
    IntervalManager() : total_available_length(0) {}

    void release_interval(int start, int end) {
        if (start >= end) return;

        auto it = intervals.lower_bound(start);

        // Merge with previous interval if overlapping or adjacent
        if (it != intervals.begin()) {
            auto prev_it = std::prev(it);
            if (prev_it->second >= start) {
                start = prev_it->first;
                end = std::max(end, prev_it->second);
                total_available_length -= prev_it->second - prev_it->first;
                intervals.erase(prev_it);
            }
        }

        // Merge with overlapping intervals
        while (it != intervals.end() && it->first <= end) {
            end = std::max(end, it->second);
            total_available_length -= it->second - it->first;
            it = intervals.erase(it);
        }

        intervals[start] = end;
        total_available_length += end - start;
    }

    void reserve_interval(int start, int end) {
        if (start >= end) return;

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
                total_available_length -= curr_end - curr_start;

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
            total_available_length += interval.second - interval.first;
        }
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

    int get_total_available_length() const {
        return total_available_length;
    }
    void print_intervals() const {
        std::ostream& out = std::cout;
        out << "Available intervals:\n";
        for (const auto& [s, e] : intervals) {
            out << "[" << s << ", " << e << ")\n";
        }
        out << "Total available length: " << total_available_length << "\n";
    }

    std::vector<std::pair<int, int>> get_intervals() const {
        return std::vector<std::pair<int, int>>(intervals.begin(), intervals.end());
    }

private:
    std::map<int, int> intervals;
    int total_available_length;
};