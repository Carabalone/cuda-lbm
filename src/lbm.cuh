#ifndef LBM_H
#define LBM_H

#include <iostream>
#include <stdint.h>
#include "defines.hpp"
#include "utility.cuh"
#include <cmath>

// for D2Q9
#ifdef D2Q9
    constexpr uint8_t dimensions = 2;
    constexpr uint8_t quadratures = 9;
    const float       cs = 1.0f / sqrt(3.0f);

    const float h_weights[]= {
          4.0f/9.0f,
          1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 
          1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f 
    };

    const int h_C[] = {
        0,  0,
        1,  0,
        0,  1,
       -1,  0,
        0, -1,
        1,  1,
       -1,  1,
       -1, -1,
        1, -1
    };
#endif

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

    __device__ static __forceinline__ int get_node_index_right(int node, int quadrature) {
        int x = node % NX;
        if (x == NX - 1) return -1;  // No right neighbor
        return node * quadratures + quadrature + 1;
    }

    __device__ static __forceinline__ int get_node_index_left(int node, int quadrature) {
        int x = node % NX;
        if (x == 0) return -1;  // No left neighbor
        return node * quadratures + quadrature - 1;
    }

    __device__ static __forceinline__ int get_node_index_up(int node, int quadrature) {
        int y = node / NX;
        if (y == 0) return -1;  // No up neighbor
        return node * quadratures + quadrature - quadratures;
    }

    __device__ static __forceinline__ int get_node_index_down(int node, int quadrature) {
        int y = node / NX;
        if (y == NY - 1) return -1; // No down neighbor
        return node * quadratures + quadrature  + quadratures;
    }

    __device__ static __forceinline__ int get_node_index_up_right(int node, int quadrature) {
        int x = node % NX;
        int y = node / NX;
        if (x == NX - 1 || y == 0) return -1; // No up-right neighbor
        return node * quadratures + quadrature - quadratures + 1;
    }

    __device__ static __forceinline__ int get_node_index_up_left(int node, int quadrature) {
        int x = node % NX;
        int y = node / NX;
        if (x == 0 || y == 0) return -1; // No up-left neighbor
        return node * quadratures + quadrature - quadratures - 1;
    }

    __device__ static __forceinline__ int get_node_index_down_right(int node, int quadrature) {
        int x = node % NX;
        int y = node / NX;
        if (x == NX - 1 || y == NY - 1) return -1; // No down-right neighbor
        return node * quadratures + quadrature + quadratures + 1;
    }

    __device__ static __forceinline__ int get_node_index_down_left(int node, int quadrature) {
        int x = node % NX;
        int y = node / NX;
        if (x == 0 || y == NY - 1) return -1; // No down-left neighbor
        return node * quadratures + quadrature + quadratures - 1;
    }

   __forceinline__ int get_velocity_index(int node) {
        //TODO
        return node;
    }

    __device__ static void equilibrium(float* f_eq, float ux, float uy, float rho, int node);

public:

    void allocate() {
        std::cout << "[LBM]: allocating\n";

        cudaMalloc((void**) &d_f,      NX * NY * quadratures * sizeof(float));
        cudaMalloc((void**) &d_f_back, NX * NY * quadratures * sizeof(float));
        cudaMalloc((void**) &d_f_eq,   NX * NY * quadratures * sizeof(float));

        cudaMalloc((void**) &d_rho,    NX * NY * sizeof(float));
        cudaMalloc((void**) &d_u,      NX * NY * dimensions * sizeof(float));

        // checkCudaErrors(cudaMemcpyToSymbol(WEIGHTS, h_weights, sizeof(float) * quadratures));
        // checkCudaErrors(cudaMemcpyToSymbol(C, h_C, sizeof(int) * dimensions * quadratures));
    }

    void free() {
        std::cout << "[LBM]: Freeing\n";

        checkCudaErrors(cudaFree(d_f));  
        checkCudaErrors(cudaFree(d_f_back));  
        checkCudaErrors(cudaFree(d_f_eq));  
        checkCudaErrors(cudaFree(d_rho));
        checkCudaErrors(cudaFree(d_u));
    }

    __device__ static void init_node(float* f, float* f_back, float* f_eq, float* rho, float* u, int node);
    __device__ static void macroscopics_node(float* f, float* rho, float* u, int node);
    __device__ static void stream_node(float* f, float* f_back, int node);

    __host__ void init();
    __host__ void macroscopics();
    __host__ void stream();

};

#endif // ! LBM_H
