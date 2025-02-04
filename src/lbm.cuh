#ifndef LBM_H
#define LBM_H

#include <iostream>
#include <stdint.h>
#include "defines.hpp"
#include "utility.cuh"

// for D2Q9
const uint8_t dimensions = 2;
const uint8_t quadratures = 9;

constexpr inline float viscosity_to_tau(float v) {
    return 3 * v + 0.5f;
}

class LBM {
private:
    float *d_f, *d_f_back;   // f, f_back: [NX][NY][Q]
    float *d_rho, *d_u;      // rho: [NX][NY], u: [NX][NY][D]
public:

    void allocate_arrays() {
        std::cout << "allocating arrays\n";

        cudaMalloc((void**) &d_f,      NX * NY * dimensions * sizeof(float));
        cudaMalloc((void**) &d_f_back, NX * NY * dimensions * sizeof(float));

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
