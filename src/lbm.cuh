#ifndef LBM_H
#define LBM_H

#include <iostream>
#include <stdint.h>
#include "defines.hpp"
#include "utility.cuh"
#include <cmath>
#include "lbm_constants.cuh"
#include "functors/defaultInit.cuh"
#include "functors/defaultBoundary.cuh"

constexpr inline float viscosity_to_tau(float v) {
    return 3 * v + 0.5f;
}

class LBM {
private:
    float *d_f, *d_f_back;   // f, f_back: [NX][NY][Q]
    float *d_rho, *d_u;      // rho: [NX][NY], u: [NX][NY][D]
    float *d_f_eq;

    __device__ static __forceinline__ int get_node_index(int node, int quadrature) {
        return node * quadratures + quadrature;
    }

   __forceinline__ int get_velocity_index(int node) {
        //TODO
        return node;
    }

public:

    void allocate() {
        std::cout << "[LBM]: allocating\n";

        cudaMalloc((void**) &d_f,      NX * NY * quadratures * sizeof(float));
        cudaMalloc((void**) &d_f_back, NX * NY * quadratures * sizeof(float));
        cudaMalloc((void**) &d_f_eq,   NX * NY * quadratures * sizeof(float));

        cudaMalloc((void**) &d_rho,    NX * NY * sizeof(float));
        cudaMalloc((void**) &d_u,      NX * NY * dimensions * sizeof(float));
    }

    void free() {
        std::cout << "[LBM]: Freeing\n";

        checkCudaErrors(cudaFree(d_f));  
        checkCudaErrors(cudaFree(d_f_back));  
        checkCudaErrors(cudaFree(d_f_eq));  
        checkCudaErrors(cudaFree(d_rho));
        checkCudaErrors(cudaFree(d_u));
    }

    __device__ static void equilibrium_node(float* f_eq, float ux, float uy, float rho, int node);
    __device__ static void init_node(float* f, float* f_back, float* f_eq, float* rho, float* u, int node);
    __device__ static void macroscopics_node(float* f, float* rho, float* u, int node);
    __device__ static void stream_node(float* f, float* f_back, int node);
    __device__ static void collide_node(float* f, float* f_eq, int node);
    __device__ static void boundaries_node(float* f, float* f_back, int node);

    __host__ void init();
    __host__ void macroscopics();
    __host__ void stream();
    __host__ void collide();
    __host__ void compute_equilibrium();
    __host__ void apply_boundaries();

};

#endif // ! LBM_H
