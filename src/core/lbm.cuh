#ifndef LBM_H
#define LBM_H

#include <iostream>
#include <stdint.h>
#include "defines.hpp"
#include "util/utility.cuh"
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>
#include "core/lbm_constants.cuh"
#include "functors/includes.cuh"
#include "assert.cuh"

#define DEBUG_NODE 5
#define VALUE_THRESHOLD 5.0f

namespace fs = std::filesystem;

class LBM {
private:
    float *d_f, *d_f_back;   // f, f_back, f_eq: [NX][NY][Q]
    float *d_rho, *d_u;      // rho: [NX][NY], u: [NX][NY][D]
    float *d_f_eq;
    float *d_force;          // force: [NX][NY][D]
    int   *d_boundary_flags; // [NX][NY]

    __device__ static __forceinline__ int get_node_index(int node, int quadrature) {
        return node * quadratures + quadrature;
    }

   __forceinline__ int get_velocity_index(int node) {
        //TODO
        return node;
    }

    template<typename BoundaryFunctor>
    void setup_boundary_flags(BoundaryFunctor boundary_func);

    template <typename Scenario>
    void send_consts() {
        checkCudaErrors(cudaMemcpyToSymbol(WEIGHTS, h_weights, sizeof(float) * quadratures));
        checkCudaErrors(cudaMemcpyToSymbol(C, h_C, sizeof(int) * dimensions * quadratures));
        checkCudaErrors(cudaMemcpyToSymbol(vis, &Scenario::viscosity, sizeof(float)));
        checkCudaErrors(cudaMemcpyToSymbol(tau, &Scenario::tau, sizeof(float)));
        checkCudaErrors(cudaMemcpyToSymbol(omega, &Scenario::omega, sizeof(float)));
        checkCudaErrors(cudaMemcpyToSymbol(OPP, h_OPP, sizeof(int) * quadratures));
    }

public:
    std::array<float, NX * NY> h_rho;
    std::array<float, NX * NY * dimensions> h_u;

    int timestep = 0, update_ts = 0;

    void allocate() {
        std::cout << "[LBM]: allocating\n";

        cudaMalloc((void**) &d_f,      NX * NY * quadratures * sizeof(float));
        cudaMalloc((void**) &d_f_back, NX * NY * quadratures * sizeof(float));
        cudaMalloc((void**) &d_f_eq,   NX * NY * quadratures * sizeof(float));

        cudaMalloc((void**) &d_rho,    NX * NY * sizeof(float));
        cudaMalloc((void**) &d_u,      NX * NY * dimensions * sizeof(float));
        
        cudaMalloc((void**) &d_force,  NX * NY * dimensions * sizeof(float));

        cudaMalloc((void**) &d_boundary_flags, NX * NY * sizeof(int));
    }

    void free() {
        std::cout << "[LBM]: Freeing\n";

        checkCudaErrors(cudaFree(d_f));  
        checkCudaErrors(cudaFree(d_f_back));  
        checkCudaErrors(cudaFree(d_f_eq));  
        checkCudaErrors(cudaFree(d_rho));
        checkCudaErrors(cudaFree(d_u));
        checkCudaErrors(cudaFree(d_force));
        checkCudaErrors(cudaFree(d_boundary_flags));
    }

    template <typename Scenario>
    void increase_ts() {
        timestep++;
        Scenario::update_ts(timestep);
    }

    void update_macroscopics() {
        int num_nodes = NX * NY;
        update_ts = timestep;

        checkCudaErrors(cudaMemcpy(h_rho.data(), d_rho, num_nodes * sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_u.data(), d_u, num_nodes * dimensions * sizeof(float), cudaMemcpyDeviceToHost));
    }

    template <typename Scenario>
    float compute_error() {
        if constexpr (Scenario::has_analytical_solution) {
            return Scenario::compute_error(*this);
        } else {
            printf("Scenario does not provide verification/validation.\n");
            return 0.0f;
        }
    }

    void save_macroscopics(int timestep) {
        int num_nodes = NX * NY;

        update_macroscopics();

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

        // printf("Saved macroscopics for timestep %d\n", timestep);
    }

    __host__ void swap_buffers() {
        float* temp;
        temp = d_f;
        d_f = d_f_back;
        d_f_back = temp;
    }

    __device__ static void equilibrium_node(float* f_eq, float ux, float uy, float rho, int node);
    __device__ static void init_node(float* f, float* f_back, float* f_eq, float* rho, float* u, int node);
    __device__ static void macroscopics_node(float* f, float* rho, float* u, float* force, int node);
    __device__ static void stream_node(float* f, float* f_back, int node);
    __device__ static void collide_node(float* f, float* f_back, float* f_eq, float* force, float* u, int node);
    __device__ static void boundaries_node(float* f, float* f_back, int node);
    __device__ static void force_node(float* force, float* u, int node);

    template<typename Scenario>
    __host__ void init();
    __host__ void macroscopics();
    __host__ void stream();
    __host__ void collide();
    __host__ void compute_equilibrium();
    template <typename Scenario>
    __host__ void apply_boundaries();
    __host__ void compute_forces();


};

#include "core/init.cuh"
#include "core/boundaries.cuh"

#endif // ! LBM_H
