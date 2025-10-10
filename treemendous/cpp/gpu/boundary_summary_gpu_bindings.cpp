// Pybind11 bindings for GPU-Accelerated Boundary Summary Manager
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// Check if CUDA is available
#ifdef WITH_CUDA
#include "boundary_summary_gpu.cu"
#define GPU_AVAILABLE true
#else
#define GPU_AVAILABLE false
#endif

namespace py = pybind11;

PYBIND11_MODULE(boundary_summary_gpu, m) {
    m.doc() = "GPU-accelerated boundary-based interval manager with summary statistics";
    
    // Compile-time GPU availability check
    m.attr("GPU_AVAILABLE") = GPU_AVAILABLE;
    
#ifdef WITH_CUDA
    // GPUInterval structure
    py::class_<GPUInterval>(m, "GPUInterval")
        .def(py::init<int, int>())
        .def_readwrite("start", &GPUInterval::start)
        .def_readwrite("end", &GPUInterval::end)
        .def("length", &GPUInterval::length)
        .def("__repr__", [](const GPUInterval& i) {
            return "<GPUInterval [" + std::to_string(i.start) + ", " + 
                   std::to_string(i.end) + ")>";
        });
    
    // GPUSummary structure
    py::class_<GPUSummary>(m, "GPUSummary")
        .def(py::init<>())
        .def_readwrite("total_free_length", &GPUSummary::total_free_length)
        .def_readwrite("total_occupied_length", &GPUSummary::total_occupied_length)
        .def_readwrite("interval_count", &GPUSummary::interval_count)
        .def_readwrite("largest_interval_length", &GPUSummary::largest_interval_length)
        .def_readwrite("largest_interval_start", &GPUSummary::largest_interval_start)
        .def_readwrite("smallest_interval_length", &GPUSummary::smallest_interval_length)
        .def_readwrite("total_gaps", &GPUSummary::total_gaps)
        .def_readwrite("earliest_start", &GPUSummary::earliest_start)
        .def_readwrite("latest_end", &GPUSummary::latest_end)
        .def_readwrite("avg_interval_length", &GPUSummary::avg_interval_length)
        .def_readwrite("avg_gap_size", &GPUSummary::avg_gap_size)
        .def_readwrite("fragmentation_index", &GPUSummary::fragmentation_index)
        .def_readwrite("utilization", &GPUSummary::utilization)
        .def("update_metrics", &GPUSummary::update_metrics)
        .def("__repr__", [](const GPUSummary& s) {
            return "<GPUSummary intervals=" + std::to_string(s.interval_count) +
                   " free=" + std::to_string(s.total_free_length) +
                   " frag=" + std::to_string(s.fragmentation_index) + ">";
        });
    
    // PerformanceStats structure
    py::class_<GPUBoundarySummaryManager::PerformanceStats>(m, "PerformanceStats")
        .def_readonly("total_operations", &GPUBoundarySummaryManager::PerformanceStats::total_operations)
        .def_readonly("gpu_operations", &GPUBoundarySummaryManager::PerformanceStats::gpu_operations)
        .def_readonly("cpu_operations", &GPUBoundarySummaryManager::PerformanceStats::cpu_operations)
        .def_readonly("gpu_utilization", &GPUBoundarySummaryManager::PerformanceStats::gpu_utilization)
        .def_readonly("gpu_memory_used", &GPUBoundarySummaryManager::PerformanceStats::gpu_memory_used)
        .def("__repr__", [](const GPUBoundarySummaryManager::PerformanceStats& s) {
            return "<PerformanceStats total_ops=" + std::to_string(s.total_operations) +
                   " gpu_ops=" + std::to_string(s.gpu_operations) +
                   " gpu_util=" + std::to_string(s.gpu_utilization * 100) + "%>";
        });
    
    // Main GPU manager class
    py::class_<GPUBoundarySummaryManager>(m, "GPUBoundarySummaryManager")
        .def(py::init<>())
        
        // Core interval operations
        .def("release_interval", &GPUBoundarySummaryManager::release_interval,
             py::arg("start"), py::arg("end"),
             "Add interval to available space")
        
        .def("reserve_interval", &GPUBoundarySummaryManager::reserve_interval,
             py::arg("start"), py::arg("end"),
             "Remove interval from available space")
        
        // Summary operations
        .def("get_summary", &GPUBoundarySummaryManager::get_summary,
             "Get summary statistics (adaptive CPU/GPU selection)")
        
        .def("compute_summary_gpu", &GPUBoundarySummaryManager::compute_summary_gpu,
             "Force GPU-accelerated summary computation")
        
        .def("compute_summary_cpu", &GPUBoundarySummaryManager::compute_summary_cpu,
             "Force CPU-based summary computation")
        
        // Advanced queries
        .def("find_best_fit_gpu", &GPUBoundarySummaryManager::find_best_fit_gpu,
             py::arg("length"), py::arg("prefer_early") = true,
             "GPU-accelerated best-fit search")
        
        // Performance and debugging
        .def("get_performance_stats", &GPUBoundarySummaryManager::get_performance_stats,
             "Get performance statistics")
        
        .def("print_info", &GPUBoundarySummaryManager::print_info,
             "Print debug information")
        
        // Python compatibility methods
        .def("get_availability_stats", [](GPUBoundarySummaryManager& self) {
            auto summary = self.get_summary();
            py::dict stats;
            stats["total_free"] = summary.total_free_length;
            stats["total_occupied"] = summary.total_occupied_length;
            stats["total_space"] = summary.total_free_length + summary.total_occupied_length;
            stats["free_chunks"] = summary.interval_count;
            stats["largest_chunk"] = summary.largest_interval_length;
            stats["avg_chunk_size"] = summary.avg_interval_length;
            stats["utilization"] = summary.utilization;
            stats["fragmentation"] = summary.fragmentation_index;
            stats["free_density"] = 1.0 - summary.utilization;
            stats["bounds"] = py::make_tuple(summary.earliest_start, summary.latest_end);
            stats["gaps"] = summary.total_gaps;
            stats["avg_gap_size"] = summary.avg_gap_size;
            return stats;
        }, "Get availability statistics (compatible format)")
        
        .def("get_total_available_length", [](GPUBoundarySummaryManager& self) {
            return self.get_summary().total_free_length;
        }, "Get total available space")
        
        .def("find_best_fit", [](GPUBoundarySummaryManager& self, int length, bool prefer_early = true) {
            auto result = self.find_best_fit_gpu(length, prefer_early);
            if (result.has_value()) {
                py::dict interval;
                interval["start"] = result->first;
                interval["end"] = result->second;
                interval["length"] = result->second - result->first;
                return interval;
            }
            return py::dict();
        }, py::arg("length"), py::arg("prefer_early") = true,
           "Find best-fit interval")
        
        .def("find_largest_available", [](GPUBoundarySummaryManager& self) {
            auto summary = self.get_summary();
            if (summary.largest_interval_length > 0) {
                py::dict interval;
                interval["start"] = summary.largest_interval_start;
                interval["end"] = summary.largest_interval_start + summary.largest_interval_length;
                interval["length"] = summary.largest_interval_length;
                return interval;
            }
            return py::dict();
        }, "Find largest available interval")
        
        .def("__repr__", [](const GPUBoundarySummaryManager&) {
            return "<GPUBoundarySummaryManager (CUDA-accelerated)>";
        });
    
    // Module-level utility functions
    m.def("get_cuda_device_info", []() {
        int device_count;
        cudaGetDeviceCount(&device_count);
        
        py::dict info;
        info["device_count"] = device_count;
        
        if (device_count > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            
            info["device_name"] = std::string(prop.name);
            info["compute_capability"] = std::to_string(prop.major) + "." + std::to_string(prop.minor);
            info["total_memory_gb"] = prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
            info["multiprocessors"] = prop.multiProcessorCount;
            info["max_threads_per_block"] = prop.maxThreadsPerBlock;
        }
        
        return info;
    }, "Get CUDA device information");
    
    m.def("benchmark_gpu_speedup", [](int num_intervals, int num_operations) {
        GPUBoundarySummaryManager manager;
        
        // Initialize with intervals
        manager.release_interval(0, num_intervals * 1000);
        
        // Perform operations
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> start_dist(0, (num_intervals - 1) * 1000);
        std::uniform_int_distribution<int> length_dist(10, 500);
        
        for (int i = 0; i < num_operations; ++i) {
            int start = start_dist(rng);
            int end = start + length_dist(rng);
            
            if (i % 2 == 0) {
                manager.reserve_interval(start, end);
            } else {
                manager.release_interval(start, end);
            }
        }
        
        // Time CPU summary
        auto cpu_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            manager.compute_summary_cpu();
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
        
        // Time GPU summary
        auto gpu_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            manager.compute_summary_gpu();
        }
        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count();
        
        py::dict result;
        result["num_intervals"] = num_intervals;
        result["num_operations"] = num_operations;
        result["cpu_time_us"] = cpu_time;
        result["gpu_time_us"] = gpu_time;
        result["speedup"] = static_cast<double>(cpu_time) / gpu_time;
        
        auto stats = manager.get_performance_stats();
        result["gpu_memory_kb"] = stats.gpu_memory_used / 1024.0;
        
        return result;
    }, py::arg("num_intervals") = 10000, py::arg("num_operations") = 5000,
       "Benchmark GPU vs CPU speedup");

#else
    // No CUDA available - provide stub implementations
    m.def("get_cuda_device_info", []() {
        py::dict info;
        info["error"] = "CUDA not available - build with WITH_CUDA=1";
        return info;
    }, "Get CUDA device information (stub)");
    
    m.def("benchmark_gpu_speedup", [](int, int) {
        py::dict result;
        result["error"] = "CUDA not available - build with WITH_CUDA=1";
        return result;
    }, py::arg("num_intervals") = 10000, py::arg("num_operations") = 5000,
       "Benchmark GPU vs CPU speedup (stub)");
#endif
}

