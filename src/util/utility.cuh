
#pragma once
#ifndef UTILITY_H
#define UTILITY_H
#include <curand_kernel.h>
#include <cstdlib>
#include <iostream>
#include <type_traits>

// #define DEBUG_KERNEL 1

// returns from [0, 1[
__device__ float inline gpu_rand(curandState &state) {
    return curand_uniform(&state);
}

__device__ float inline gpu_rand(curandState &state, float min, float max) {
    return (min + (max - min) * curand_uniform(&state));
}

inline double random_double() {
    return rand() / (RAND_MAX + 1.0);
}

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

/* from LBM Principles and Practive accompanying code */
inline void __getLastCudaError(const char *const errorMessage, const char *const file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s(%d): [%d] %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

// template <typename T>
// __global__ 
// void sum_arrays_kernel(float* arr1, float* arr2, float* res, size_t size);

// template <typename T>
// inline void sum_arrays(float* arr1, float* arr2, float* res, size_t size) {
//     dim3 threads(256);
//     dim3 blocks((size+255) / 256);

//     sum_arrays_kernel<<<threads, blocks>>>(arr1, arr2, res, size);
//     checkCudaErrors(cudaDeviceSynchronize());
// }

#ifdef DEBUG_KERNEL
  #define DPRINTF(fmt, ...) printf(fmt, __VA_ARGS__)
#else
  #define DPRINTF(fmt, ...)
#endif

namespace utl {
    
template <typename T>
__device__ __forceinline__
bool in_range(T var, T min, T max) {
    return var > min && var < max;
}

}

#endif // UTILITY_H
