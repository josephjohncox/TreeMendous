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
    const py::object domain = py::module_::import("treemendous.domain");
    const py::object span_type = domain.attr("Span");
    const py::object interval_result_type = domain.attr("IntervalResult");
    const py::object mutation_result_type = domain.attr("MutationResult");
    const auto mutation_result = [span_type, mutation_result_type](
                                     const MutationDelta& delta) -> py::object {
        py::tuple changed(delta.changed.size());
        for (std::size_t index = 0; index < delta.changed.size(); ++index) {
            const auto& [start, end] = delta.changed[index];
            changed[index] = span_type(start, end);
        }
        return mutation_result_type(
            changed, delta.changed_length, delta.fully_covered);
    };

    py::class_<IntervalManager>(m, "IntervalManager")
        .def(py::init<>())
        .def("set_managed_domain", [](IntervalManager& manager,
                                       const py::iterable& raw_spans) {
            std::vector<std::pair<Coordinate, Coordinate>> spans;
            for (py::handle item : raw_spans) {
                if (!PySequence_Check(item.ptr())) {
                    throw py::type_error("managed domain spans must be pairs");
                }
                py::sequence pair = py::reinterpret_borrow<py::sequence>(item);
                if (py::len(pair) != 2) {
                    throw py::type_error("managed domain spans must be pairs");
                }
                spans.emplace_back(
                    checked_integer(pair[0], "managed domain start"),
                    checked_integer(pair[1], "managed domain end")
                );
            }
            manager.set_managed_domain(spans);
        })
        .def("release_interval", [](IntervalManager& manager, py::handle start, py::handle end) {
            manager.release_interval(checked_integer(start, "start"), checked_integer(end, "end"));
        })
        .def("reserve_interval", [](IntervalManager& manager, py::handle start, py::handle end) {
            manager.reserve_interval(checked_integer(start, "start"), checked_integer(end, "end"));
        })
        .def("release_with_delta", [mutation_result](IntervalManager& manager,
                                                       py::handle start,
                                                       py::handle end) {
            const Coordinate checked_start = checked_integer(start, "start");
            const Coordinate checked_end = checked_integer(end, "end");
            const MutationDelta delta = manager.preview_release_delta(
                checked_start, checked_end);
            py::object result = mutation_result(delta);
            manager.release_interval(checked_start, checked_end);
            return result;
        })
        .def("reserve_with_delta", [mutation_result](IntervalManager& manager,
                                                       py::handle start,
                                                       py::handle end,
                                                       bool require_covered) {
            const Coordinate checked_start = checked_integer(start, "start");
            const Coordinate checked_end = checked_integer(end, "end");
            const MutationDelta delta = manager.preview_reserve_delta(
                checked_start, checked_end, require_covered);
            py::object result = mutation_result(delta);
            if (!require_covered || delta.fully_covered) {
                manager.reserve_interval(checked_start, checked_end);
            }
            return result;
        })
        .def("find_interval", [](const IntervalManager& manager,
                                  py::handle point,
                                  py::handle length) -> py::object {
            const auto result = manager.find_interval(
                checked_integer(point, "point"), checked_integer(length, "length"));
            if (result.first == result.second) return py::none();
            return py::cast(result);
        })
        .def("allocate_interval", [interval_result_type](IntervalManager& manager,
                                                          py::handle point,
                                                          py::handle length,
                                                          const py::object& not_after) -> py::object {
            const Coordinate checked_point = checked_integer(point, "point");
            const Measure checked_length = checked_integer(length, "length");
            const bool bounded = !not_after.is_none();
            const Coordinate limit = bounded
                ? checked_integer(not_after, "not_after")
                : 0;
            const auto allocation = manager.find_interval(
                checked_point, checked_length);
            if (allocation.first == allocation.second ||
                (bounded && allocation.second > limit)) {
                return py::none();
            }
            py::object result = interval_result_type(
                allocation.first, allocation.second);
            manager.reserve_interval(allocation.first, allocation.second);
            return result;
        })
        .def("find_overlapping_intervals", [](const IntervalManager& manager,
                                               py::handle start, py::handle end) {
            return manager.find_overlapping_intervals(
                checked_integer(start, "start"), checked_integer(end, "end"));
        })
        .def("get_total_available_length", &IntervalManager::get_total_available_length)
        .def("get_interval_count", &IntervalManager::get_interval_count)
        .def("get_largest_available_length", &IntervalManager::get_largest_available_length)
        .def("print_intervals", &IntervalManager::print_intervals)
        .def("get_intervals", &IntervalManager::get_intervals);
    m.attr("IntervalManager").attr("_treemendous_authoritative_geometry") = true;

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
