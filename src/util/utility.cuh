
#pragma once
#ifndef UTILITY_H
#define UTILITY_H
#include <curand_kernel.h>
#include <cstdlib>
#include <iostream>
#include <type_traits>

// Verbose diagnostics (host + `__device__` printf sites that use `DPRINTF` / `LBM_LOG`).
// Define exactly once in `defines.hpp` (or via `-DLBM_DEBUG` on the compile line).
#ifdef LBM_DEBUG
  #define LBM_LOG(fmt, ...) printf(fmt, __VA_ARGS__)
#else
  #define LBM_LOG(fmt, ...)
#endif

// Historical name used across kernels; keep as an alias so call sites stay stable.
#define DPRINTF LBM_LOG

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

namespace utl {
    
template <typename T>
__device__ __forceinline__
bool in_range(T var, T min, T max) {
    return var > min && var < max;
}

__device__ __forceinline__
float clamp_ramp_up_linear(float u_max, int ramp_up_time, int timestep) {

    return ramp_up_time > 0 ?
        u_max * fminf(1.0f, ((float) timestep) / ramp_up_time) :
        u_max;
}

__device__ __forceinline__
float clamp_ramp_up_tanh(float u_max, int ramp_up_time, int timestep) {

    if (ramp_up_time <= 0) {
        return u_max;
    }

    float progress = static_cast<float>(timestep) / ramp_up_time;
    float scaled_progress = 12.0f * (progress - 0.5f);
    return fmaxf(u_max * 0.5f * (tanhf(scaled_progress) + 1.0f), 0.001f);

}

__device__ __forceinline__
float clamp(float val, float min, float max) {
    return fminf(fmaxf(val, min), max);    
}

__device__ __forceinline__
float clamp_abs(float val, float lim) {
    return fminf(fmaxf(val, -lim), lim);    
}

}

#endif // UTILITY_H
