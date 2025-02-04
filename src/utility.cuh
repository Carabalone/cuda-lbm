
#pragma once
#ifndef UTILITY_H
#define UTILITY_H
#include <curand_kernel.h>
#include <cstdlib>
#include <iostream>

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

#endif // UTILITY_H
