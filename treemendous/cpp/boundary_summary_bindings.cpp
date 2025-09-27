// Pybind11 bindings for Boundary-Based Summary Interval Manager
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "boundary_summary_optimized.cpp"

namespace py = pybind11;

PYBIND11_MODULE(boundary_summary, m) {
    m.doc() = "High-performance boundary-based interval manager with summary statistics";
    
    // BoundarySummary structure
    py::class_<BoundarySummary>(m, "BoundarySummary")
        .def_readwrite("total_free_length", &BoundarySummary::total_free_length)
        .def_readwrite("total_occupied_length", &BoundarySummary::total_occupied_length)
        .def_readwrite("interval_count", &BoundarySummary::interval_count)
        .def_readwrite("largest_interval_length", &BoundarySummary::largest_interval_length)
        .def_readwrite("largest_interval_start", &BoundarySummary::largest_interval_start)
        .def_readwrite("smallest_interval_length", &BoundarySummary::smallest_interval_length)
        .def_readwrite("avg_interval_length", &BoundarySummary::avg_interval_length)
        .def_readwrite("total_gaps", &BoundarySummary::total_gaps)
        .def_readwrite("avg_gap_size", &BoundarySummary::avg_gap_size)
        .def_readwrite("fragmentation_index", &BoundarySummary::fragmentation_index)
        .def_readwrite("earliest_start", &BoundarySummary::earliest_start)
        .def_readwrite("latest_end", &BoundarySummary::latest_end)
        .def_readwrite("utilization", &BoundarySummary::utilization)
        .def("update_metrics", &BoundarySummary::update_metrics);
    
    // AvailabilityStats structure
    py::class_<BoundarySummaryManager::AvailabilityStats>(m, "AvailabilityStats")
        .def_readwrite("total_free", &BoundarySummaryManager::AvailabilityStats::total_free)
        .def_readwrite("total_occupied", &BoundarySummaryManager::AvailabilityStats::total_occupied)
        .def_readwrite("total_space", &BoundarySummaryManager::AvailabilityStats::total_space)
        .def_readwrite("free_chunks", &BoundarySummaryManager::AvailabilityStats::free_chunks)
        .def_readwrite("largest_chunk", &BoundarySummaryManager::AvailabilityStats::largest_chunk)
        .def_readwrite("avg_chunk_size", &BoundarySummaryManager::AvailabilityStats::avg_chunk_size)
        .def_readwrite("utilization", &BoundarySummaryManager::AvailabilityStats::utilization)
        .def_readwrite("fragmentation", &BoundarySummaryManager::AvailabilityStats::fragmentation)
        .def_readwrite("free_density", &BoundarySummaryManager::AvailabilityStats::free_density)
        .def_readwrite("bounds", &BoundarySummaryManager::AvailabilityStats::bounds)
        .def_readwrite("gaps", &BoundarySummaryManager::AvailabilityStats::gaps)
        .def_readwrite("avg_gap_size", &BoundarySummaryManager::AvailabilityStats::avg_gap_size);
    
    // PerformanceStats structure
    py::class_<BoundarySummaryManager::PerformanceStats>(m, "PerformanceStats")
        .def_readwrite("operation_count", &BoundarySummaryManager::PerformanceStats::operation_count)
        .def_readwrite("cache_hits", &BoundarySummaryManager::PerformanceStats::cache_hits)
        .def_readwrite("cache_hit_rate", &BoundarySummaryManager::PerformanceStats::cache_hit_rate)
        .def_readwrite("implementation", &BoundarySummaryManager::PerformanceStats::implementation)
        .def_readwrite("interval_count", &BoundarySummaryManager::PerformanceStats::interval_count);
    
    // BoundarySummaryManager class
    py::class_<BoundarySummaryManager>(m, "BoundarySummaryManager")
        .def(py::init<>())
        
        // Core interval operations
        .def("release_interval", &BoundarySummaryManager::release_interval,
             py::arg("start"), py::arg("end"),
             "Add interval to available space")
        .def("reserve_interval", &BoundarySummaryManager::reserve_interval,
             py::arg("start"), py::arg("end"),
             "Remove interval from available space")
        .def("find_interval", &BoundarySummaryManager::find_interval,
             py::arg("start"), py::arg("length"),
             "Find available interval of given length")
        .def("get_intervals", &BoundarySummaryManager::get_intervals,
             "Get all available intervals")
        .def("get_total_available_length", &BoundarySummaryManager::get_total_available_length,
             "Get total available space")
        
        // Advanced query operations
        .def("find_best_fit", &BoundarySummaryManager::find_best_fit,
             py::arg("length"), py::arg("prefer_early") = true,
             "Find best-fit interval with preference")
        .def("find_largest_available", &BoundarySummaryManager::find_largest_available,
             "Find largest available interval")
        
        // Summary statistics
        .def("get_summary", &BoundarySummaryManager::get_summary,
             "Get comprehensive boundary summary")
        .def("get_availability_stats", &BoundarySummaryManager::get_availability_stats,
             "Get availability statistics in standard format")
        .def("get_performance_stats", &BoundarySummaryManager::get_performance_stats,
             "Get performance and caching statistics")
        
        // Utilities
        .def("print_intervals", &BoundarySummaryManager::print_intervals,
             "Print intervals with summary information");
    
    // Module-level utilities
    m.def("benchmark_boundary_summary", []() {
        BoundarySummaryManager manager;
        manager.release_interval(0, 100000);
        
        // Generate random operations
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> op_dist(0, 1);
        std::uniform_int_distribution<int> start_dist(0, 90000);
        std::uniform_int_distribution<int> length_dist(1, 1000);
        
        const int num_operations = 10000;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_operations; ++i) {
            std::string op = (op_dist(rng) == 0) ? "reserve" : "release";
            int start = start_dist(rng);
            int end = start + length_dist(rng);
            
            if (op == "reserve") {
                manager.reserve_interval(start, end);
            } else {
                manager.release_interval(start, end);
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // Benchmark summary access
        auto summary_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000; ++i) {
            auto stats = manager.get_availability_stats();
        }
        auto summary_end = std::chrono::high_resolution_clock::now();
        auto summary_duration = std::chrono::duration_cast<std::chrono::microseconds>(summary_end - summary_start);
        
        py::dict result;
        result["operations"] = num_operations;
        result["time_microseconds"] = duration.count();
        result["ops_per_second"] = static_cast<double>(num_operations) / (duration.count() / 1000000.0);
        result["summary_time_microseconds"] = summary_duration.count();
        result["avg_summary_time_us"] = static_cast<double>(summary_duration.count()) / 1000.0;
        
        auto summary = manager.get_summary();
        result["final_intervals"] = summary.interval_count;
        result["fragmentation"] = summary.fragmentation_index;
        result["utilization"] = summary.utilization;
        
        auto perf = manager.get_performance_stats();
        result["cache_hit_rate"] = perf.cache_hit_rate;
        
        return result;
    }, "Benchmark boundary summary manager performance");
    
    // Version and metadata
    m.attr("__version__") = "0.2.0";
    m.attr("__author__") = "Joseph Cox";
    m.attr("__description__") = "Boundary-based interval manager with summary statistics";
}
