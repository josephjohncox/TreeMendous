// Pybind11 bindings for C++ Treap implementation
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "treap.cpp"  // Include the treap implementation

namespace py = pybind11;

PYBIND11_MODULE(treap, m) {
    m.doc() = "High-performance C++ Treap implementation for interval trees";
    
    // TreapNode class
    py::class_<TreapNode>(m, "TreapNode")
        .def(py::init<int, int, double>(), py::arg("start"), py::arg("end"), py::arg("priority") = -1.0)
        .def_readwrite("start", &TreapNode::start)
        .def_readwrite("end", &TreapNode::end)
        .def_readwrite("priority", &TreapNode::priority)
        .def_readwrite("height", &TreapNode::height)
        .def_readwrite("subtree_size", &TreapNode::subtree_size)
        .def_readwrite("total_length", &TreapNode::total_length)
        .def("length", &TreapNode::length)
        .def("update_stats", &TreapNode::update_stats)
        .def_static("get_height", &TreapNode::get_height)
        .def_static("get_size", &TreapNode::get_size);
    
    // TreapStatistics structure
    py::class_<IntervalTreap::TreapStatistics>(m, "TreapStatistics")
        .def_readwrite("size", &IntervalTreap::TreapStatistics::size)
        .def_readwrite("height", &IntervalTreap::TreapStatistics::height)
        .def_readwrite("expected_height", &IntervalTreap::TreapStatistics::expected_height)
        .def_readwrite("balance_factor", &IntervalTreap::TreapStatistics::balance_factor)
        .def_readwrite("total_length", &IntervalTreap::TreapStatistics::total_length)
        .def_readwrite("avg_interval_length", &IntervalTreap::TreapStatistics::avg_interval_length);
    
    // IntervalTreap class
    py::class_<IntervalTreap>(m, "IntervalTreap")
        .def(py::init<>(), "Create treap with random seed")
        .def(py::init<unsigned int>(), py::arg("seed"), "Create treap with fixed seed")
        
        // Core interval operations
        .def("release_interval", &IntervalTreap::release_interval,
             py::arg("start"), py::arg("end"),
             "Add interval to available space")
        .def("reserve_interval", &IntervalTreap::reserve_interval,
             py::arg("start"), py::arg("end"),
             "Remove interval from available space")
        .def("find_interval", &IntervalTreap::find_interval,
             py::arg("start"), py::arg("length"),
             "Find available interval of given length")
        .def("get_intervals", &IntervalTreap::get_intervals,
             "Get all available intervals")
        .def("get_total_available_length", &IntervalTreap::get_total_available_length,
             "Get total available space")
        
        // Tree properties
        .def("get_tree_size", &IntervalTreap::get_tree_size,
             "Get number of intervals in treap")
        .def("get_height", &IntervalTreap::get_height,
             "Get tree height")
        .def("get_expected_height", &IntervalTreap::get_expected_height,
             "Get expected height for current size")
        
        // Treap-specific operations
        .def("sample_random_interval", &IntervalTreap::sample_random_interval,
             "Sample random interval uniformly")
        .def("split", &IntervalTreap::split,
             py::arg("key"),
             "Split treap at given key into two treaps")
        .def("find_overlapping_intervals", &IntervalTreap::find_overlapping_intervals,
             py::arg("start"), py::arg("end"),
             "Find all intervals overlapping with range")
        
        // Analysis and verification
        .def("verify_treap_properties", &IntervalTreap::verify_treap_properties,
             "Verify BST and heap properties")
        .def("get_statistics", &IntervalTreap::get_statistics,
             "Get comprehensive treap statistics")
        .def("print_tree", &IntervalTreap::print_tree,
             "Print tree structure");
    
    // HighPerformanceTreap class
    py::class_<HighPerformanceTreap, IntervalTreap>(m, "HighPerformanceTreap")
        .def(py::init<>())
        .def("bulk_insert", &HighPerformanceTreap::bulk_insert,
             py::arg("intervals"),
             "Insert multiple intervals efficiently")
        .def("bulk_delete", &HighPerformanceTreap::bulk_delete,
             py::arg("intervals"),
             "Delete multiple intervals efficiently");
    
    // Module-level utilities
    m.def("test_treap_performance", []() {
        IntervalTreap treap(42);  // Fixed seed
        
        // Performance test
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < 10000; ++i) {
            treap.release_interval(i * 10, i * 10 + 5);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        auto stats = treap.get_statistics();
        
        py::dict result;
        result["operations"] = 10000;
        result["time_microseconds"] = duration.count();
        result["ops_per_second"] = 10000.0 / (duration.count() / 1000000.0);
        result["height"] = stats.height;
        result["expected_height"] = stats.expected_height;
        result["balance_factor"] = stats.balance_factor;
        
        return result;
    }, "Run performance test on C++ treap");
    
    // Version and metadata
    m.attr("__version__") = "0.2.0";
    m.attr("__author__") = "Joseph Cox";
    m.attr("__description__") = "High-performance randomized interval trees (treaps)";
}
