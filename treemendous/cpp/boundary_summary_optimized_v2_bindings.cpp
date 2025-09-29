// Python bindings for optimized BoundarySummaryManager
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "boundary_summary_optimized_v2.cpp"

namespace py = pybind11;

PYBIND11_MODULE(boundary_summary_optimized, m) {
    m.doc() = "Optimized C++ Boundary Summary Manager with boost::flat_map and small vector optimizations";

    // Don't expose IntervalResult or BoundarySummary - both cause conflicts
    // Users should import from original module if needed:
    // from treemendous.cpp.boundary_summary import IntervalResult, BoundarySummary
    // The optimized version returns the same types, so no need to re-register them

    py::class_<BoundarySummaryManagerOptimized::AvailabilityStats>(m, "AvailabilityStats")
        .def(py::init<>())
        .def_readwrite("total_free", &BoundarySummaryManagerOptimized::AvailabilityStats::total_free)
        .def_readwrite("total_occupied", &BoundarySummaryManagerOptimized::AvailabilityStats::total_occupied)
        .def_readwrite("total_space", &BoundarySummaryManagerOptimized::AvailabilityStats::total_space)
        .def_readwrite("free_chunks", &BoundarySummaryManagerOptimized::AvailabilityStats::free_chunks)
        .def_readwrite("largest_chunk", &BoundarySummaryManagerOptimized::AvailabilityStats::largest_chunk)
        .def_readwrite("avg_chunk_size", &BoundarySummaryManagerOptimized::AvailabilityStats::avg_chunk_size)
        .def_readwrite("utilization", &BoundarySummaryManagerOptimized::AvailabilityStats::utilization)
        .def_readwrite("fragmentation", &BoundarySummaryManagerOptimized::AvailabilityStats::fragmentation)
        .def_readwrite("free_density", &BoundarySummaryManagerOptimized::AvailabilityStats::free_density)
        .def_readwrite("bounds", &BoundarySummaryManagerOptimized::AvailabilityStats::bounds)
        .def_readwrite("gaps", &BoundarySummaryManagerOptimized::AvailabilityStats::gaps)
        .def_readwrite("avg_gap_size", &BoundarySummaryManagerOptimized::AvailabilityStats::avg_gap_size);

    py::class_<BoundarySummaryManagerOptimized::PerformanceStats>(m, "PerformanceStats")
        .def(py::init<>())
        .def_readwrite("operation_count", &BoundarySummaryManagerOptimized::PerformanceStats::operation_count)
        .def_readwrite("cache_hits", &BoundarySummaryManagerOptimized::PerformanceStats::cache_hits)
        .def_readwrite("cache_hit_rate", &BoundarySummaryManagerOptimized::PerformanceStats::cache_hit_rate)
        .def_readwrite("implementation", &BoundarySummaryManagerOptimized::PerformanceStats::implementation)
        .def_readwrite("interval_count", &BoundarySummaryManagerOptimized::PerformanceStats::interval_count);

    py::class_<BoundarySummaryManagerOptimized>(m, "BoundarySummaryManager")
        .def(py::init<>())
        .def("release_interval", &BoundarySummaryManagerOptimized::release_interval,
             py::arg("start"), py::arg("end"))
        .def("reserve_interval", &BoundarySummaryManagerOptimized::reserve_interval,
             py::arg("start"), py::arg("end"))
        .def("find_interval", &BoundarySummaryManagerOptimized::find_interval,
             py::arg("start"), py::arg("length"))
        .def("find_best_fit", &BoundarySummaryManagerOptimized::find_best_fit,
             py::arg("length"), py::arg("prefer_early") = true)
        .def("find_largest_available", &BoundarySummaryManagerOptimized::find_largest_available)
        .def("get_summary", &BoundarySummaryManagerOptimized::get_summary)
        .def("get_availability_stats", &BoundarySummaryManagerOptimized::get_availability_stats)
        .def("get_total_available_length", &BoundarySummaryManagerOptimized::get_total_available_length)
        .def("get_intervals", &BoundarySummaryManagerOptimized::get_intervals)
        .def("get_performance_stats", &BoundarySummaryManagerOptimized::get_performance_stats)
        .def("print_intervals", &BoundarySummaryManagerOptimized::print_intervals);

    // Expose optimization flags
    m.attr("USE_FLAT_MAP") = TREE_MENDOUS_USE_FLAT_MAP;
    m.attr("USE_SMALL_VECTOR") = TREE_MENDOUS_USE_SMALL_VECTOR;
    m.attr("USE_SIMD") = TREE_MENDOUS_USE_SIMD;
    m.attr("PREALLOCATE_VECTORS") = TREE_MENDOUS_PREALLOCATE_VECTORS;
    
    m.def("get_optimization_info", []() {
        py::dict info;
        info["flat_map"] = TREE_MENDOUS_USE_FLAT_MAP;
        info["small_vector"] = TREE_MENDOUS_USE_SMALL_VECTOR;
        info["simd"] = TREE_MENDOUS_USE_SIMD;
        info["preallocate"] = TREE_MENDOUS_PREALLOCATE_VECTORS;
        return info;
    }, "Get dictionary of enabled optimizations");
}
