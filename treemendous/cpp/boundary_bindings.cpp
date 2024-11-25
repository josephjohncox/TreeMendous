// Pybind11 bindings for IntervalManager
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "boundary_ic.cpp"  // Include the IntervalManager implementation
#include "boundary.cpp"  // Include the IntervalManager implementation

namespace py = pybind11;

PYBIND11_MODULE(boundary, m) {
    py::class_<IntervalManager>(m, "IntervalManager")
        .def(py::init<>())
        .def("release_interval", &IntervalManager::release_interval)
        .def("reserve_interval", &IntervalManager::reserve_interval)
        .def("find_interval", &IntervalManager::find_interval)
        .def("get_total_available_length", &IntervalManager::get_total_available_length)
        .def("print_intervals", &IntervalManager::print_intervals)
        .def("get_intervals", &IntervalManager::get_intervals);
    py::class_<ICIntervalManager>(m, "ICIntervalManager")
        .def(py::init<>())
        .def("release_interval", &ICIntervalManager::release_interval)
        .def("reserve_interval", &ICIntervalManager::reserve_interval)
        .def("find_interval", &ICIntervalManager::find_interval)
        .def("get_total_available_length", &ICIntervalManager::get_total_available_length)
        .def("print_intervals", &ICIntervalManager::print_intervals)
        .def("get_intervals", &ICIntervalManager::get_intervals);
}
