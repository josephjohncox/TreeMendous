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
    
    // IntervalResult structure
    py::class_<IntervalResult>(m, "IntervalResult")
        .def(py::init<int, int>(), "Construct IntervalResult with start and end")
        .def(py::init<int, int, int>(), "Construct IntervalResult with start, end, and length")
        .def_readwrite("start", &IntervalResult::start)
        .def_readwrite("end", &IntervalResult::end)
        .def_readwrite("length", &IntervalResult::length)
        .def("__repr__", [](const IntervalResult &ir) {
            return "IntervalResult(start=" + std::to_string(ir.start) +
                   ", end=" + std::to_string(ir.end) +
                   ", length=" + std::to_string(ir.length) + ")";
        });
    
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
        .def("find_best_fit", [](BoundarySummaryManager &self, int length, bool prefer_early) -> py::object {
            auto result = self.find_best_fit(length, prefer_early);
            if (result) {
                return py::cast(*result);
            } else {
                return py::none();
            }
        }, py::arg("length"), py::arg("prefer_early") = true,
             "Find best-fit interval with preference")
        .def("find_largest_available", [](BoundarySummaryManager &self) -> py::object {
            auto result = self.find_largest_available();
            if (result) {
                return py::cast(*result);
            } else {
                return py::none();
            }
        }, "Find largest available interval")
        
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
    
}
