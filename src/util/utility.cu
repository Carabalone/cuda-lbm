#include "utility.cuh"

// do not use this function, use CheckCudaErrors from utility.cuh instead.
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
                  << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// template <typename T>
__global__ 
void sum_arrays_kernel(float* arr1, float* arr2, float* result, size_t size) {
    // static_assert(std::is_arithmetic<T>::value, "Type must be numeric");

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= size)
        return;

    result[idx] = arr1[idx] + arr2[idx];
}
