// Python bindings for optimized IntervalManager
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "boundary_optimized.cpp"

namespace py = pybind11;

PYBIND11_MODULE(boundary_optimized, m) {
    m.doc() = "Optimized C++ Interval Manager with boost::flat_map, SIMD, and small vector optimizations";

    py::class_<IntervalManagerOptimized>(m, "IntervalManager")
        .def(py::init<>())
        .def("release_interval", &IntervalManagerOptimized::release_interval,
             py::arg("start"), py::arg("end"),
             "Release an interval, making it available")
        .def("reserve_interval", &IntervalManagerOptimized::reserve_interval,
             py::arg("start"), py::arg("end"),
             "Reserve an interval, making it unavailable")
        .def("find_interval", &IntervalManagerOptimized::find_interval,
             py::arg("point"), py::arg("length"),
             "Find an available interval of given length starting at or after point")
        .def("get_total_available_length", &IntervalManagerOptimized::get_total_available_length,
             "Get total available length across all intervals")
        .def("get_intervals", &IntervalManagerOptimized::get_intervals,
             "Get all available intervals as list of (start, end) tuples")
        .def("print_intervals", &IntervalManagerOptimized::print_intervals,
             "Print all intervals and optimization status");
    
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
