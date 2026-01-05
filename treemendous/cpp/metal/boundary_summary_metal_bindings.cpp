// Pybind11 bindings for Metal-accelerated Boundary Summary Manager
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <chrono>
#include <algorithm>
#include <iterator>

// Include the C++ header (implementation is in the .mm file)
#include "boundary_summary_metal.h"

namespace py = pybind11;

PYBIND11_MODULE(boundary_summary_metal, m) {
    m.doc() = "Metal-accelerated boundary-based interval manager for macOS";
    
    // Check Metal availability at module level
    m.attr("METAL_AVAILABLE") = true;  // If we compiled, Metal is available
    
    // MetalSummary structure
    py::class_<MetalSummary>(m, "MetalSummary")
        .def(py::init<>())
        .def_readwrite("total_free_length", &MetalSummary::total_free_length)
        .def_readwrite("total_occupied_length", &MetalSummary::total_occupied_length)
        .def_readwrite("interval_count", &MetalSummary::interval_count)
        .def_readwrite("largest_interval_length", &MetalSummary::largest_interval_length)
        .def_readwrite("largest_interval_start", &MetalSummary::largest_interval_start)
        .def_readwrite("smallest_interval_length", &MetalSummary::smallest_interval_length)
        .def_readwrite("total_gaps", &MetalSummary::total_gaps)
        .def_readwrite("earliest_start", &MetalSummary::earliest_start)
        .def_readwrite("latest_end", &MetalSummary::latest_end)
        .def_readwrite("avg_interval_length", &MetalSummary::avg_interval_length)
        .def_readwrite("avg_gap_size", &MetalSummary::avg_gap_size)
        .def_readwrite("fragmentation_index", &MetalSummary::fragmentation_index)
        .def_readwrite("utilization", &MetalSummary::utilization)
        .def("update_metrics", &MetalSummary::update_metrics)
        .def("__repr__", [](const MetalSummary& s) {
            return "<MetalSummary intervals=" + std::to_string(s.interval_count) +
                   " free=" + std::to_string(s.total_free_length) +
                   " frag=" + std::to_string(s.fragmentation_index) + ">";
        });
    
    // PerformanceStats structure
    py::class_<MetalBoundarySummaryManager::PerformanceStats>(m, "PerformanceStats")
        .def_readonly("total_operations", &MetalBoundarySummaryManager::PerformanceStats::total_operations)
        .def_readonly("gpu_operations", &MetalBoundarySummaryManager::PerformanceStats::gpu_operations)
        .def_readonly("gpu_utilization", &MetalBoundarySummaryManager::PerformanceStats::gpu_utilization)
        .def_readonly("gpu_memory_used", &MetalBoundarySummaryManager::PerformanceStats::gpu_memory_used)
        .def("__repr__", [](const MetalBoundarySummaryManager::PerformanceStats& s) {
            return "<PerformanceStats total_ops=" + std::to_string(s.total_operations) +
                   " gpu_ops=" + std::to_string(s.gpu_operations) +
                   " gpu_util=" + std::to_string(s.gpu_utilization * 100) + "%>";
        });
    
    // Main Metal manager class
    py::class_<MetalBoundarySummaryManager>(m, "MetalBoundarySummaryManager")
        .def(py::init<>())
        
        // Core interval operations
        .def("release_interval", &MetalBoundarySummaryManager::release_interval,
             py::arg("start"), py::arg("end"),
             "Add interval to available space")
        
        .def("reserve_interval", &MetalBoundarySummaryManager::reserve_interval,
             py::arg("start"), py::arg("end"),
             "Remove interval from available space")
        
        // Batch operations (GPU-optimized - single Python call)
        .def("batch_reserve", [](MetalBoundarySummaryManager& self, const py::list& intervals) {
            std::vector<std::pair<int, int>> cpp_intervals;
            cpp_intervals.reserve(intervals.size());
            for (auto item : intervals) {
                auto interval = item.cast<py::tuple>();
                cpp_intervals.emplace_back(interval[0].cast<int>(), interval[1].cast<int>());
            }
            self.batch_reserve(cpp_intervals);
        }, py::arg("intervals"),
           "Reserve multiple intervals in a single call (GPU-optimized)")
        
        .def("batch_release", [](MetalBoundarySummaryManager& self, const py::list& intervals) {
            std::vector<std::pair<int, int>> cpp_intervals;
            cpp_intervals.reserve(intervals.size());
            for (auto item : intervals) {
                auto interval = item.cast<py::tuple>();
                cpp_intervals.emplace_back(interval[0].cast<int>(), interval[1].cast<int>());
            }
            self.batch_release(cpp_intervals);
        }, py::arg("intervals"),
           "Release multiple intervals in a single call (GPU-optimized)")

        // Summary operations
        .def("get_summary", &MetalBoundarySummaryManager::get_summary,
             "Get summary statistics (adaptive CPU/GPU selection)")
        
        .def("compute_summary_gpu", &MetalBoundarySummaryManager::compute_summary_gpu,
             "Force Metal-accelerated summary computation")
        
        .def("compute_summary_cpu", &MetalBoundarySummaryManager::compute_summary_cpu,
             "Force CPU-based summary computation")
        
        // Advanced queries
        .def("find_best_fit_gpu", &MetalBoundarySummaryManager::find_best_fit_gpu,
             py::arg("length"), py::arg("prefer_early") = true,
             "Metal-accelerated best-fit search")
        
        // Performance and debugging
        .def("get_performance_stats", &MetalBoundarySummaryManager::get_performance_stats,
             "Get performance statistics")
        
        .def("print_info", &MetalBoundarySummaryManager::print_info,
             "Print debug information")
        
        // Python compatibility methods
        .def("get_availability_stats", [](MetalBoundarySummaryManager& self) {
            auto summary = self.get_summary();
            py::dict stats;
            stats["total_free"] = summary.total_free_length;
            stats["total_occupied"] = summary.total_occupied_length;
            stats["total_space"] = summary.total_free_length + summary.total_occupied_length;
            stats["free_chunks"] = summary.interval_count;
            stats["largest_chunk"] = summary.largest_interval_length;
            stats["avg_chunk_size"] = summary.avg_interval_length;
            stats["utilization"] = summary.utilization;
            stats["fragmentation"] = summary.fragmentation_index;
            stats["free_density"] = 1.0 - summary.utilization;
            stats["bounds"] = py::make_tuple(summary.earliest_start, summary.latest_end);
            stats["gaps"] = summary.total_gaps;
            stats["avg_gap_size"] = summary.avg_gap_size;
            return stats;
        }, "Get availability statistics (compatible format)")
        
        .def("get_total_available_length", [](MetalBoundarySummaryManager& self) {
            return self.get_summary().total_free_length;
        }, "Get total available space")
        
        .def("get_intervals", [](MetalBoundarySummaryManager& self) {
            auto intervals = self.get_intervals();
            py::list result;
            for (const auto& [start, end] : intervals) {
                result.append(py::make_tuple(start, end));
            }
            return result;
        }, "Get all available intervals")
        
        .def("find_interval", [](MetalBoundarySummaryManager& self, int start, int length) -> py::object {
            if (length <= 0) {
                return py::none();
            }
            
            auto intervals = self.get_intervals();
            if (intervals.empty()) {
                return py::none();
            }
            
            auto it = std::lower_bound(
                intervals.begin(),
                intervals.end(),
                start,
                [](const std::pair<int, int>& interval, int value) {
                    return interval.first < value;
                }
            );
            
            bool found = false;
            int result_start = 0;
            int result_end = 0;
            
            if (it != intervals.end()) {
                int s = it->first;
                int e = it->second;
                if (s <= start && start < e && e - start >= length) {
                    result_start = start;
                    result_end = start + length;
                    found = true;
                } else if (s > start && e - s >= length) {
                    result_start = s;
                    result_end = s + length;
                    found = true;
                }
            }
            
            if (!found && it != intervals.begin()) {
                auto prev = std::prev(it);
                int s = prev->first;
                int e = prev->second;
                if (s <= start && start < e && e - start >= length) {
                    result_start = start;
                    result_end = start + length;
                    found = true;
                } else if (start < s && e - s >= length) {
                    result_start = s;
                    result_end = s + length;
                    found = true;
                }
            }
            
            if (found) {
                return py::make_tuple(result_start, result_end);
            }
            
            return py::none();
        }, py::arg("start"), py::arg("length"),
           "Find interval of given length (compatibility method)")
        
        .def("find_best_fit", [](MetalBoundarySummaryManager& self, int length, bool prefer_early = true) -> py::object {
            if (length <= 0) {
                return py::none();
            }
            
            if (prefer_early) {
                // Earliest-fit path to match CPU boundary summary semantics
                auto intervals = self.get_intervals();
                for (const auto& [start, end] : intervals) {
                    if (end - start >= length) {
                        return py::make_tuple(start, start + length);
                    }
                }
                return py::none();
            }
            
            auto result = self.find_best_fit_gpu(length, false);
            if (result.has_value()) {
                return py::make_tuple(result->first, result->first + length);
            }
            return py::none();
        }, py::arg("length"), py::arg("prefer_early") = true,
           "Find best-fit interval")
        
        .def("find_largest_available", [](MetalBoundarySummaryManager& self) -> py::object {
            auto summary = self.get_summary();
            if (summary.largest_interval_length > 0) {
                return py::make_tuple(
                    summary.largest_interval_start,
                    summary.largest_interval_start + summary.largest_interval_length
                );
            }
            return py::none();
        }, "Find largest available interval")
        
        .def("__repr__", [](const MetalBoundarySummaryManager&) {
            return "<MetalBoundarySummaryManager (Metal Performance Shaders)>";
        });
    
    // Module-level utility functions
    m.def("get_metal_device_info", []() {
        auto info_map = get_metal_device_info();
        py::dict info;
        for (const auto& [key, value] : info_map) {
            info[key.c_str()] = value;
        }
        return info;
    }, "Get Metal device information");
    
    m.def("benchmark_metal_speedup", [](int num_intervals, int num_operations) {
        MetalBoundarySummaryManager manager;
        
        // Initialize with intervals
        manager.release_interval(0, num_intervals * 1000);
        
        // Perform operations
        std::random_device rd;
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> start_dist(0, (num_intervals - 1) * 1000);
        std::uniform_int_distribution<int> length_dist(10, 500);
        
        for (int i = 0; i < num_operations; ++i) {
            int start = start_dist(rng);
            int end = start + length_dist(rng);
            
            if (i % 2 == 0) {
                manager.reserve_interval(start, end);
            } else {
                manager.release_interval(start, end);
            }
        }
        
        // Time CPU summary
        auto cpu_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            manager.compute_summary_cpu();
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
        
        // Time Metal summary
        auto metal_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            manager.compute_summary_gpu();
        }
        auto metal_end = std::chrono::high_resolution_clock::now();
        auto metal_time = std::chrono::duration_cast<std::chrono::microseconds>(metal_end - metal_start).count();
        
        py::dict result;
        result["num_intervals"] = num_intervals;
        result["num_operations"] = num_operations;
        result["cpu_time_us"] = cpu_time;
        result["metal_time_us"] = metal_time;
        result["speedup"] = static_cast<double>(cpu_time) / metal_time;
        
        auto stats = manager.get_performance_stats();
        result["metal_memory_kb"] = stats.gpu_memory_used / 1024.0;
        
        return result;
    }, py::arg("num_intervals") = 10000, py::arg("num_operations") = 5000,
       "Benchmark Metal vs CPU speedup");
}
