// The CUDA implementation and pybind11 module are compiled as one NVCC
// translation unit in boundary_summary_gpu.cu. Keeping this source marker makes
// the sdist layout explicit without textually including a .cu file or emitting
// duplicate definitions.
