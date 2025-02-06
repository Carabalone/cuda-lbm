#ifndef LBM_H
#define LBM_H

#include <iostream>
#include <stdint.h>
#include "defines.hpp"
#include "utility.cuh"

// for D2Q9
#ifdef D2Q9
    const uint8_t dimensions = 2;
    const uint8_t quadratures = 9;

    __constant__ static const float[quadratures] WEIGHTS = {
          4.0f/9.0f,
          1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 
          1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f 
    };

    __constant__ static const int[quadratures * dimensions] C = {
        0,  0,
        1,  0,
        0,  1,
       -1,  0,
        0, -1,
        1,  1,
       -1,  1,
       -1, -1,
        1, -1
    }
#endif

constexpr inline float viscosity_to_tau(float v) {
    return 3 * v + 0.5f;
}

class LBM {
private:
    float *d_f, *d_f_back;   // f, f_back: [NX][NY][Q]
    float *d_rho, *d_u;      // rho: [NX][NY], u: [NX][NY][D]
    float *d_feq

    __host__ __device__ __forceinline__ inline int get_node_index(int node, int quadrature) {
        return node * quadratures + quadrature;
    }

    __host__ __device__ __forceinline__ inline int get_velocity_index(int node) {

    }

    __host__ __device__ void equilibrium(float* f_eq, int node, float ux, float uy, float rho);

    __host__ __device__ void init(float* f, float* f_back, float* f_eq, float* rho, float* u);

public:

    void allocate_arrays() {
        std::cout << "allocating arrays\n";

        cudaMalloc((void**) &d_f,      NX * NY * quadratures * sizeof(float));
        cudaMalloc((void**) &d_f_back, NX * NY * quadratures * sizeof(float));
        cudaMalloc((void**) &d_f_eq,   NX * NY * quadratures * sizeof(float));

        cudaMalloc((void**) &d_rho,    NX * NY * sizeof(float));
        cudaMalloc((void**) &d_u,      NX * NY * dimensions * sizeof(float));
    }

    void free_arrays() {
        std::cout << "freeing arrays\n";

        checkCudaErrors(cudaFree(d_f));  
        checkCudaErrors(cudaFree(d_f_back));  
        checkCudaErrors(cudaFree(d_rho));
        checkCudaErrors(cudaFree(d_u));
    }
};

#endif // ! LBM_H
