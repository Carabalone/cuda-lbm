#ifndef LBM_H
#define LBM_H

#include <iostream>
#include <stdint.h>
#include "defines.hpp"
#include "utility.cuh"
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>
#include "lbm_constants.cuh"
#include "./functors/defaultInit.cuh"
#include "./functors/defaultBoundary.cuh"
#include "./functors/cylinderBoundary.cuh"
#include "./functors/zouHeInflow.cuh"
#include "./functors/zeroGradientOutflow.cuh"

namespace fs = std::filesystem;

constexpr inline float viscosity_to_tau(float v) {
    return 3 * v + 0.5f;
}

class LBM {
private:
    float *d_f, *d_f_back;   // f, f_back, f_eq: [NX][NY][Q]
    float *d_rho, *d_u;      // rho: [NX][NY], u: [NX][NY][D]
    float *d_f_eq;
    int   *d_boundary_flags; // [NX][NY]

    __device__ static __forceinline__ int get_node_index(int node, int quadrature) {
        return node * quadratures + quadrature;
    }

   __forceinline__ int get_velocity_index(int node) {
        //TODO
        return node;
    }

    void setup_boundary_flags();

public:

    void allocate() {
        std::cout << "[LBM]: allocating\n";

        cudaMalloc((void**) &d_f,      NX * NY * quadratures * sizeof(float));
        cudaMalloc((void**) &d_f_back, NX * NY * quadratures * sizeof(float));
        cudaMalloc((void**) &d_f_eq,   NX * NY * quadratures * sizeof(float));

        cudaMalloc((void**) &d_rho,    NX * NY * sizeof(float));
        cudaMalloc((void**) &d_u,      NX * NY * dimensions * sizeof(float));

        cudaMalloc((void**) &d_boundary_flags, NX * NY * sizeof(int));
    }

    void free() {
        std::cout << "[LBM]: Freeing\n";

        checkCudaErrors(cudaFree(d_f));  
        checkCudaErrors(cudaFree(d_f_back));  
        checkCudaErrors(cudaFree(d_f_eq));  
        checkCudaErrors(cudaFree(d_rho));
        checkCudaErrors(cudaFree(d_u));
        checkCudaErrors(cudaFree(d_boundary_flags));
    }

    void save_macroscopics(int timestep) {
        int num_nodes = NX * NY;
        
        std::vector<float> h_rho(num_nodes);
        std::vector<float> h_u(2 * num_nodes);

        checkCudaErrors(cudaMemcpy(h_rho.data(), d_rho, num_nodes * sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_u.data(), d_u, 2 * num_nodes * sizeof(float), cudaMemcpyDeviceToHost));

        std::ostringstream rho_filename, vel_filename;
        rho_filename << "output/density/density_" << timestep << ".bin";
        vel_filename << "output/velocity/velocity_" << timestep << ".bin";

        if (!fs::is_directory("output/density") || !fs::exists("output/density"))
            fs::create_directory("output/density");

        std::ofstream rho_file(rho_filename.str(), std::ios::out | std::ios::binary);
        if (!rho_file) {
            printf("Error: Could not open file %s for writing.\n", rho_filename.str().c_str());
            return;
        }
        rho_file.write(reinterpret_cast<const char*>(h_rho.data()), num_nodes * sizeof(float));
        rho_file.close();

        if (!fs::is_directory("output/velocity") || !fs::exists("output/velocity"))
            fs::create_directory("output/velocity");
        std::ofstream vel_file(vel_filename.str(), std::ios::out | std::ios::binary);
        if (!vel_file) {
            printf("Error: Could not open file %s for writing.\n", vel_filename.str().c_str());
            return;
        }
        vel_file.write(reinterpret_cast<const char*>(h_u.data()), 2 * num_nodes * sizeof(float));
        vel_file.close();

        printf("Saved macroscopics for timestep %d\n", timestep);
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
