// Binding-compatible parity alias for the canonical boundary manager.
#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "boundary_optimized.cpp"

namespace py = pybind11;

namespace {
Coordinate checked_integer(const py::handle& value, const char* name) {
    if (!PyLong_Check(value.ptr()) || PyBool_Check(value.ptr())) {
        throw py::type_error(std::string(name) + " must be an integer");
    }
    const long long converted = PyLong_AsLongLong(value.ptr());
    if (converted == -1 && PyErr_Occurred()) throw py::error_already_set();
    return static_cast<Coordinate>(converted);
}
}  // namespace

PYBIND11_MODULE(boundary_optimized, m) {
    m.doc() = "Parity alias for the canonical checked C++ boundary manager";

    py::class_<IntervalManagerOptimized>(m, "IntervalManager")
        .def(py::init<>())
        .def("release_interval", [](IntervalManagerOptimized& manager, py::handle start, py::handle end) {
            manager.release_interval(checked_integer(start, "start"), checked_integer(end, "end"));
        })
        .def("reserve_interval", [](IntervalManagerOptimized& manager, py::handle start, py::handle end) {
            manager.reserve_interval(checked_integer(start, "start"), checked_integer(end, "end"));
        })
        .def("find_interval", [](const IntervalManagerOptimized& manager, py::handle point, py::handle length) -> py::object {
            const auto result = manager.find_interval(
                checked_integer(point, "point"), checked_integer(length, "length"));
            if (result.first == result.second) return py::none();
            return py::cast(result);
        })
        .def("get_total_available_length", &IntervalManagerOptimized::get_total_available_length)
        .def("get_intervals", &IntervalManagerOptimized::get_intervals)
        .def("print_intervals", &IntervalManagerOptimized::print_intervals);

    // Retained only for compatibility; every specialization is deliberately off.
    m.attr("USE_FLAT_MAP") = false;
    m.attr("USE_SMALL_VECTOR") = false;
    m.attr("USE_SIMD") = false;
    m.attr("PREALLOCATE_VECTORS") = false;
    m.def("get_optimization_info", []() {
        py::dict info;
        info["parity_alias"] = true;
        info["flat_map"] = false;
        info["small_vector"] = false;
        info["simd"] = false;
        info["preallocate"] = false;
        return info;
    });
}
#endif  // __has_include(<pybind11/pybind11.h>)
