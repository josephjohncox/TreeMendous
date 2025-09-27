// Pybind11 bindings for Summary-Enhanced Interval Managers
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Include both simple and summary implementations
#include "boundary.cpp"  // Simple IntervalManager
#ifdef WITH_IC_MANAGER
#include "boundary_ic.cpp"  // Simple ICIntervalManager  
#endif

#include "summary_boundary.cpp"  // Summary-enhanced versions
#ifdef WITH_IC_MANAGER
#include "summary_boundary_ic.cpp"
#endif

namespace py = pybind11;

PYBIND11_MODULE(boundary, m) {
    m.doc() = "Tree-Mendous C++ Interval Tree Implementations with Summary Statistics";
    
    // ==== SIMPLE IMPLEMENTATIONS ====
    
    // Basic IntervalManager (std::map based)
    py::class_<IntervalManager>(m, "SimpleIntervalManager", 
        "Basic interval manager using std::map - optimized for simplicity")
        .def(py::init<>())
        .def("release_interval", &IntervalManager::release_interval,
             "Add interval to available space")
        .def("reserve_interval", &IntervalManager::reserve_interval,
             "Remove interval from available space")
        .def("find_interval", &IntervalManager::find_interval,
             "Find available interval of given length starting from point")
        .def("get_total_available_length", &IntervalManager::get_total_available_length,
             "Get total available space")
        .def("print_intervals", &IntervalManager::print_intervals,
             "Print all available intervals")
        .def("get_intervals", &IntervalManager::get_intervals,
             "Get list of all available intervals");

#ifdef WITH_IC_MANAGER
    // Basic ICIntervalManager (Boost ICL based)
    py::class_<ICIntervalManager>(m, "SimpleICIntervalManager",
        "Interval manager using Boost Interval Containers - optimized for set operations")
        .def(py::init<>())
        .def("release_interval", &ICIntervalManager::release_interval)
        .def("reserve_interval", &ICIntervalManager::reserve_interval)
        .def("find_interval", &ICIntervalManager::find_interval)
        .def("get_total_available_length", &ICIntervalManager::get_total_available_length)
        .def("print_intervals", &ICIntervalManager::print_intervals)
        .def("get_intervals", &ICIntervalManager::get_intervals);
#endif

    // ==== SUMMARY-ENHANCED IMPLEMENTATIONS ====
    
    // TreeSummary structure
    py::class_<TreeSummary>(m, "TreeSummary", 
        "Comprehensive aggregate statistics for interval trees")
        .def_readwrite("total_free_length", &TreeSummary::total_free_length)
        .def_readwrite("total_occupied_length", &TreeSummary::total_occupied_length) 
        .def_readwrite("contiguous_count", &TreeSummary::contiguous_count)
        .def_readwrite("largest_free_length", &TreeSummary::largest_free_length)
        .def_readwrite("largest_free_start", &TreeSummary::largest_free_start)
        .def_readwrite("earliest_free_start", &TreeSummary::earliest_free_start)
        .def_readwrite("latest_free_end", &TreeSummary::latest_free_end)
        .def_readwrite("avg_free_length", &TreeSummary::avg_free_length)
        .def_readwrite("free_density", &TreeSummary::free_density)
        .def_readwrite("utilization", &TreeSummary::utilization)
        .def_readwrite("fragmentation", &TreeSummary::fragmentation);
    
    // AvailabilityStats structure for detailed metrics
    py::class_<SummaryIntervalManager::AvailabilityStats>(m, "AvailabilityStats",
        "Detailed availability and utilization statistics")
        .def_readwrite("total_free", &SummaryIntervalManager::AvailabilityStats::total_free)
        .def_readwrite("total_occupied", &SummaryIntervalManager::AvailabilityStats::total_occupied)
        .def_readwrite("total_space", &SummaryIntervalManager::AvailabilityStats::total_space)
        .def_readwrite("free_chunks", &SummaryIntervalManager::AvailabilityStats::free_chunks)
        .def_readwrite("largest_chunk", &SummaryIntervalManager::AvailabilityStats::largest_chunk)
        .def_readwrite("avg_chunk_size", &SummaryIntervalManager::AvailabilityStats::avg_chunk_size)
        .def_readwrite("utilization", &SummaryIntervalManager::AvailabilityStats::utilization)
        .def_readwrite("fragmentation", &SummaryIntervalManager::AvailabilityStats::fragmentation)
        .def_readwrite("free_density", &SummaryIntervalManager::AvailabilityStats::free_density)
        .def_readwrite("bounds", &SummaryIntervalManager::AvailabilityStats::bounds);
        
    // Summary-Enhanced IntervalManager (std::map based)
    py::class_<SummaryIntervalManager>(m, "IntervalManager",
        "🌟 Summary-enhanced interval manager with comprehensive aggregate statistics")
        .def(py::init<>())
        .def("release_interval", &SummaryIntervalManager::release_interval,
             "Add interval to available space with summary update")
        .def("reserve_interval", &SummaryIntervalManager::reserve_interval,
             "Remove interval from available space with summary update")
        .def("find_interval", &SummaryIntervalManager::find_interval,
             "Find available interval using summary-optimized search")
        .def("find_best_fit", &SummaryIntervalManager::find_best_fit,
             py::arg("length"), py::arg("prefer_early") = true,
             "Find best-fit interval using summary statistics for rapid pruning")
        .def("find_largest_available", &SummaryIntervalManager::find_largest_available,
             "Find largest available interval using O(1) summary lookup")
        .def("get_summary", &SummaryIntervalManager::get_summary,
             "Get comprehensive tree summary with aggregate statistics")
        .def("get_availability_stats", &SummaryIntervalManager::get_availability_stats,
             "Get detailed availability and utilization statistics")
        .def("get_total_available_length", &SummaryIntervalManager::get_total_available_length,
             "Get total available space (legacy compatibility)")
        .def("print_intervals", &SummaryIntervalManager::print_intervals,
             "Print intervals with summary statistics")
        .def("get_intervals", &SummaryIntervalManager::get_intervals,
             "Get list of all available intervals");

#ifdef WITH_IC_MANAGER
    // Summary-Enhanced ICIntervalManager (Boost ICL based)
    py::class_<SummaryICIntervalManager>(m, "ICIntervalManager",
        "🌟 Summary-enhanced interval manager using Boost ICL with aggregate statistics")
        .def(py::init<>())
        .def("release_interval", &SummaryICIntervalManager::release_interval)
        .def("reserve_interval", &SummaryICIntervalManager::reserve_interval)
        .def("find_interval", &SummaryICIntervalManager::find_interval)
        .def("find_best_fit", &SummaryICIntervalManager::find_best_fit,
             py::arg("length"), py::arg("prefer_early") = true,
             "Find best-fit interval using summary statistics")
        .def("find_largest_available", &SummaryICIntervalManager::find_largest_available,
             "Find largest available interval using O(1) summary lookup")
        .def("get_summary", &SummaryICIntervalManager::get_summary,
             "Get comprehensive tree summary with aggregate statistics")
        .def("get_availability_stats", &SummaryICIntervalManager::get_availability_stats,
             "Get detailed availability and utilization statistics")
        .def("get_total_available_length", &SummaryICIntervalManager::get_total_available_length)
        .def("print_intervals", &SummaryICIntervalManager::print_intervals)
        .def("get_intervals", &SummaryICIntervalManager::get_intervals);
#endif
    
    // ==== MODULE-LEVEL UTILITIES ====
    
    m.def("get_implementation_info", []() {
        py::dict info;
        info["simple_implementations"] = py::list();
        info["summary_implementations"] = py::list();
        
        py::cast(info["simple_implementations"]).attr("append")("SimpleIntervalManager");
#ifdef WITH_IC_MANAGER
        py::cast(info["simple_implementations"]).attr("append")("SimpleICIntervalManager");
#endif
        
        py::cast(info["summary_implementations"]).attr("append")("IntervalManager");
#ifdef WITH_IC_MANAGER
        py::cast(info["summary_implementations"]).attr("append")("ICIntervalManager");
#endif
        
        info["features"] = py::dict();
        py::cast(info["features"])["summary_statistics"] = true;
        py::cast(info["features"])["best_fit_queries"] = true;
        py::cast(info["features"])["fragmentation_analysis"] = true;
        py::cast(info["features"])["utilization_monitoring"] = true;
        
        return info;
    }, "Get information about available implementations and features");
    
    // Version information
    m.attr("__version__") = "0.2.0";
    m.attr("__author__") = "Joseph Cox";
    m.attr("__description__") = "High-performance interval trees with summary statistics";
}
