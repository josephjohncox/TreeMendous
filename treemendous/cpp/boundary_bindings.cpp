// Pybind11 bindings for IntervalManager
#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#ifdef WITH_IC_MANAGER
#include "boundary_ic.cpp"
#endif
#include "boundary.cpp"

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

PYBIND11_MODULE(boundary, m) {
    py::class_<IntervalManager>(m, "IntervalManager")
        .def(py::init<>())
        .def("release_interval", [](IntervalManager& manager, py::handle start, py::handle end) {
            manager.release_interval(checked_integer(start, "start"), checked_integer(end, "end"));
        })
        .def("reserve_interval", [](IntervalManager& manager, py::handle start, py::handle end) {
            manager.reserve_interval(checked_integer(start, "start"), checked_integer(end, "end"));
        })
        .def("find_interval", [](const IntervalManager& manager, py::handle point, py::handle length) -> py::object {
            const auto result = manager.find_interval(
                checked_integer(point, "point"), checked_integer(length, "length"));
            if (result.first == result.second) return py::none();
            return py::cast(result);
        })
        .def("get_total_available_length", &IntervalManager::get_total_available_length)
        .def("print_intervals", &IntervalManager::print_intervals)
        .def("get_intervals", &IntervalManager::get_intervals);

#ifdef WITH_IC_MANAGER
    py::class_<ICIntervalManager>(m, "ICIntervalManager")
        .def(py::init<>())
        .def("release_interval", &ICIntervalManager::release_interval)
        .def("reserve_interval", &ICIntervalManager::reserve_interval)
        .def("find_interval", &ICIntervalManager::find_interval)
        .def("get_total_available_length", &ICIntervalManager::get_total_available_length)
        .def("print_intervals", &ICIntervalManager::print_intervals)
        .def("get_intervals", &ICIntervalManager::get_intervals);
#endif
}
#endif  // __has_include(<pybind11/pybind11.h>)
