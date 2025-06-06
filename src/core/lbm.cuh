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
#include "IBM/IBMManager.cuh"

#define DEBUG
#define DEBUG_KERNEL
#define DEBUG_NODE 5
#define VALUE_THRESHOLD 5.0f

namespace fs = std::filesystem;

struct MomentInfo {
    float rho_avg_norm;
    float j_avg_norm;
    float pi_avg_norm;
};

extern __device__ MomentInfo d_moment_avg;   // used for adaptive relaxation

template <int dim>
class LBM {
private:
    float *d_f, *d_f_back;   // f, f_back, f_eq: [NX][NY][Q]
    float *d_rho, *d_u;      // rho: [NX][NY], u: [NX][NY][D]
    float *d_f_eq;
    float *d_force;            // force: [NX][NY][D]
    int   *d_boundary_flags;   // [NX][NY]
    float *d_u_uncorrected;

    float *d_pi_mag; // pi_mag: [NX][NY] -> used for adaptive relaxation

    template<typename BoundaryFunctor>
    void setup_boundary_flags(BoundaryFunctor boundary_func);

    void matmul(const float* M, const float* N, float* P, int size) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                P[i*size + j] = 0.0f;
            }
        }
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < size; k++) {
                    P[i*size + j] += M[i*size + k] * N[k*size + j];
                }
            }
        }
    }

    template <typename Scenario>
    void send_consts() {
        checkCudaErrors(cudaMemcpyToSymbol(vis, &Scenario::viscosity, sizeof(float)));
        checkCudaErrors(cudaMemcpyToSymbol(tau, &Scenario::tau,       sizeof(float)));
        checkCudaErrors(cudaMemcpyToSymbol(omega, &Scenario::omega,   sizeof(float)));
        
        checkCudaErrors(cudaMemcpyToSymbol(C, h_C,             sizeof(int) * dimensions * quadratures));
        checkCudaErrors(cudaMemcpyToSymbol(OPP, h_OPP,         sizeof(int) * quadratures));
        checkCudaErrors(cudaMemcpyToSymbol(WEIGHTS, h_weights, sizeof(float) * quadratures));
        checkCudaErrors(cudaMemcpyToSymbol(M, h_M,             sizeof(float) * quadratures * quadratures));
        checkCudaErrors(cudaMemcpyToSymbol(M_inv, h_M_inv,     sizeof(float) * quadratures * quadratures));
        checkCudaErrors(cudaMemcpyToSymbol(S, Scenario::S,     sizeof(float) * quadratures));

        // LBM_ASSERT(fabs(Scenario::S[7] - Scenario::omega) < 1e-6, 
        //            "MRT relaxation value S[7] does not match scenario viscosity");

        // LBM_ASSERT(fabs(Scenario::S[8] - Scenario::omega) < 1e-6,
        //            "MRT relaxation value S[8] does not match scenario viscosity"); 

    }

public:
    std::vector<float> h_rho;
    std::vector<float> h_u;
    
    IBMManager<dim> IBM;

    int timestep = 0, update_ts = 0;

    template <typename Scenario>
    void allocate() {
        std::cout << "[LBM]: allocating\n";
        // for 2D, NZ is 1 by default.
        constexpr int num_nodes = NX * NY * NZ;

        h_rho.resize(num_nodes);
        h_u.resize(num_nodes * dimensions);

    #ifdef CSOA
        constexpr int num_clusters = (num_nodes + cluster_size - 1) / cluster_size;
        constexpr int alloc_nodes = num_clusters * cluster_size;
    #else
        constexpr int alloc_nodes = num_nodes;
    #endif

        cudaMalloc((void**) &d_f,      alloc_nodes * quadratures * sizeof(float));
        cudaMalloc((void**) &d_f_back, alloc_nodes * quadratures * sizeof(float));
        cudaMalloc((void**) &d_f_eq,   alloc_nodes * quadratures * sizeof(float));

        cudaMalloc((void**) &d_rho,    NX * NY * NZ * sizeof(float));
        cudaMalloc((void**) &d_u,      NX * NY * NZ * dimensions * sizeof(float));
        cudaMalloc((void**) &d_u_uncorrected,      NX * NY * NZ * dimensions * sizeof(float));
        
        cudaMalloc((void**) &d_force,  NX * NY * NZ * dimensions * sizeof(float));

        cudaMalloc((void**) &d_boundary_flags, NX * NY * NZ * sizeof(int));

        // cudaMalloc((void**) &d_moment_avg, sizeof(MomentInfo));
        cudaMalloc((void**) &d_pi_mag, NX * NY * NZ * sizeof(float));

        Scenario::add_bodies();
        IBM.init_and_dispatch(Scenario::IBM_bodies);
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
        // checkCudaErrors(cudaFree(d_moment_avg));
        checkCudaErrors(cudaFree(d_pi_mag));
        checkCudaErrors(cudaFree(d_u_uncorrected));
    }

    template <typename Scenario>
    void increase_ts() {
        timestep++;
        Scenario::update_ts(timestep);
    }

    void update_macroscopics() {
        int num_nodes = NX * NY * NZ;
        update_ts = timestep;

        checkCudaErrors(cudaMemcpy(h_rho.data(), d_rho, num_nodes * sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_u.data(), d_u, num_nodes * dimensions * sizeof(float), cudaMemcpyDeviceToHost));
    }

    float* get_rho() {
        return d_rho;
    }
    float* get_u() {
        return d_u;
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

    void save_midplane_slice(int timestep) {
        // Mid-plane index
        constexpr int z_mid = NZ / 2;
        constexpr int num_nodes_2d = NX * NY;

        // Host buffers for the slice
        std::vector<float> h_rho_slice(num_nodes_2d);
        std::vector<float> h_u_slice(num_nodes_2d * 3); // 3 for ux, uy, uz

        // Download full arrays from device
        std::vector<float> h_rho_full(NX * NY * NZ);
        std::vector<float> h_u_full(NX * NY * NZ * 3);

        checkCudaErrors(cudaMemcpy(
            h_rho_full.data(), d_rho, NX * NY * NZ * sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(
            h_u_full.data(), d_u, NX * NY * NZ * 3 * sizeof(float), cudaMemcpyDeviceToHost));

        // SoA offsets
        size_t u_stride = NX * NY * NZ;

        // Extract the mid-plane
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                int idx_2d = y * NX + x;
                int idx_3d = z_mid * NX * NY + y * NX + x;

                h_rho_slice[idx_2d] = h_rho_full[idx_3d];

                h_u_slice[3 * idx_2d + 0] = h_u_full[0 * u_stride + idx_3d]; // ux
                h_u_slice[3 * idx_2d + 1] = h_u_full[1 * u_stride + idx_3d]; // uy
                h_u_slice[3 * idx_2d + 2] = h_u_full[2 * u_stride + idx_3d]; // uz
            }
        }

        // Save to file
        namespace fs = std::filesystem;
        if (!fs::is_directory("output/midplane_density") || !fs::exists("output/midplane_density"))
            fs::create_directory("output/midplane_density");
        if (!fs::is_directory("output/midplane_velocity") || !fs::exists("output/midplane_velocity"))
            fs::create_directory("output/midplane_velocity");

        std::ostringstream rho_filename, vel_filename;
        rho_filename << "output/midplane_density/density_" << timestep << ".bin";
        vel_filename << "output/midplane_velocity/velocity_" << timestep << ".bin";

        std::ofstream rho_file(rho_filename.str(), std::ios::out | std::ios::binary);
        if (!rho_file) {
            printf("Error: Could not open file %s for writing.\n", rho_filename.str().c_str());
            return;
        }
        rho_file.write(reinterpret_cast<const char*>(h_rho_slice.data()), num_nodes_2d * sizeof(float));
        rho_file.close();

        std::ofstream vel_file(vel_filename.str(), std::ios::out | std::ios::binary);
        if (!vel_file) {
            printf("Error: Could not open file %s for writing.\n", vel_filename.str().c_str());
            return;
        }
        vel_file.write(reinterpret_cast<const char*>(h_u_slice.data()), num_nodes_2d * 3 * sizeof(float));
        vel_file.close();
    }

    void save_vtk(int timestep) {
        update_macroscopics();

        std::string output_dir = "output/vtk";
        try {
            if (!fs::is_directory(output_dir) || !fs::exists(output_dir)) {
                fs::create_directories(output_dir);
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Filesystem error creating directory " << output_dir << ": " << e.what() << std::endl;
            return;
        }

        std::ostringstream filename_stream;
        filename_stream << output_dir << "/sim_data_" << std::setw(6) << std::setfill('0') << timestep << ".vti";
        std::string filename = filename_stream.str();

        std::ofstream vtk_file(filename, std::ios::out | std::ios::binary);
        if (!vtk_file) {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return;
        }

        std::cout << "[VTK Export] Saving data for timestep " << timestep << " to " << filename << std::endl;

        vtk_file << "<?xml version=\"1.0\"?>\n";
        vtk_file << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
        vtk_file << "  <ImageData WholeExtent=\"0 " << NX - 1 << " 0 " << NY - 1 << " 0 " << NZ - 1
                 << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";
        vtk_file << "    <Piece Extent=\"0 " << NX - 1 << " 0 " << NY - 1 << " 0 " << NZ - 1 << "\">\n";
        vtk_file << "      <PointData Scalars=\"Density\" Vectors=\"Velocity\">\n";

        vtk_file << "        <DataArray type=\"Float32\" Name=\"Density\" format=\"appended\" offset=\"0\"/>\n";
        vtk_file << "        <DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"appended\" offset=\""
                 << h_rho.size() * sizeof(float) << "\"/>\n";

        vtk_file << "      </PointData>\n";
        vtk_file << "      <CellData>\n";
        vtk_file << "      </CellData>\n";
        vtk_file << "    </Piece>\n";
        vtk_file << "  </ImageData>\n";
        vtk_file << "  <AppendedData encoding=\"raw\">\n";
        vtk_file << "   _";

        std::vector<float> u_3d_buffer;
        u_3d_buffer.reserve(static_cast<size_t>(NX) * NY * NZ * 3);

        for (int k = 0; k < NZ; ++k) {
            for (int j = 0; j < NY; ++j) {
                for (int i = 0; i < NX; ++i) {
                    int node_idx = k * (NX * NY) + j * NX + i;
                    u_3d_buffer.push_back(h_u[get_vec_index(node_idx, 0)]);
                    u_3d_buffer.push_back(h_u[get_vec_index(node_idx, 1)]);
                    if constexpr (dim == 3) {
                        u_3d_buffer.push_back(h_u[get_vec_index(node_idx, 2)]);
                    } else { // dim == 2
                        u_3d_buffer.push_back(0.0f); // Pad Z component
                    }
                }
            }
        }

        uint64_t density_bytes = h_rho.size() * sizeof(float);
        vtk_file.write(reinterpret_cast<const char*>(&density_bytes), sizeof(uint64_t));
        vtk_file.write(reinterpret_cast<const char*>(h_rho.data()), density_bytes);

        uint64_t velocity_bytes = u_3d_buffer.size() * sizeof(float);
        vtk_file.write(reinterpret_cast<const char*>(&velocity_bytes), sizeof(uint64_t));
        vtk_file.write(reinterpret_cast<const char*>(u_3d_buffer.data()), velocity_bytes);

        vtk_file << "\n  </AppendedData>\n";
        vtk_file << "</VTKFile>\n";

        vtk_file.close();
    }

    __host__ void swap_buffers() {
        float* temp;
        temp = d_f;
        d_f = d_f_back;
        d_f_back = temp;
    }

    void ibm_step() {
        IBM.multi_direct(d_rho, d_u, d_force);
    }

    template <typename Scenario>
    void reset_forces();

    __device__ static void equilibrium_node(float* f_eq, float ux, float uy, float uz, float rho, int node);
    __device__ static void init_node(float* f, float* f_back, float* f_eq, float* rho, float* u, int node);
    __device__ static void stream_node(float* f, float* f_back, int node);
    template <typename CollisionOp>
    __device__ static void collide_node(float* f, float* f_eq, float* u, float* force, int node, int q);
    __device__ static void boundaries_node(float* f, float* f_back, int node);
    __device__ static void force_node(float* force, float* u, int node);

    template<typename Scenario>
    __host__ void init();
    __host__ void uncorrected_macroscopics();
    __host__ void correct_macroscopics();
    __host__ void stream();
    template <typename CollisionOp>
    __host__ void collide();
    __host__ void compute_equilibrium();
    template <typename Scenario>
    __host__ void apply_boundaries();
    __host__ void compute_forces();

    ~LBM() {
        free();
    }
};

#include "core/init/init.cuh"
#include "core/streaming/streaming.cuh"
#include "core/macroscopics/macroscopics.cuh"
#include "core/equilibrium/equilibrium.cuh"
#include "core/boundaries/boundaries.cuh"
#include "core/collision/collision.cuh"

#endif // ! LBM_H
