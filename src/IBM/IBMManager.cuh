#ifndef IBMMANAGER_H
#define IBMMANAGER_H
#include "IBM/IBMBody.cuh"
#include "IBM/IBM_impl.cuh"

#include <vector>
#define ITER_MAX 3

// Forward declarations of kernels
template <int dim>
__global__
void interpolate_velocities_kernel(float* lag_points, float* lag_u, float* lag_rho, int num_points, float* eul_u, float* eul_rho);

template <int dim>
__global__
void spread_forces_kernel(float* lag_points, float* lag_force, int num_points, float* eul_force);

template <int dim>
__global__
void accumulate_forces_kernel(float* eul_force_total, float* eul_force_iter);

template <int dim>
__global__
void correct_velocities_kernel(float* eul_u, float* eul_force_iter, float* eul_rho);

template <int dim>
__global__
void compute_lagrangian_kernel(float* lag_points, float* lag_u, float* lag_force, int num_points, float* lag_rho);

template <int dim>
struct IBMManager {
    float* d_lag_points;        // Boundary point coordinates [dim*num_points]
    float* d_lag_u;             // Velocities at boundary points [dim*num_points]
    float* d_lag_rho;           // Densities at boundary points [num_points]
    float* d_lag_force;         // Forces at boundary points [dim*num_points]
    
    float* d_eul_force_acc;     // Accumulated forces on grid [dim*NX*NY*NZ]
    
    float* d_eul_u_prev;        // Previous iteration velocities [dim*NX*NY*NZ]
    float* d_eul_force_iter;    // Current iteration forces [dim*NX*NY*NZ]
    
    std::vector<IBMBody> h_bodies;
    int num_points;
    int num_bodies;
    int iter_count;

    IBMManager() : d_lag_points(nullptr), d_lag_u(nullptr), d_lag_rho(nullptr),
                   d_lag_force(nullptr), d_eul_force_acc(nullptr),
                   d_eul_u_prev(nullptr), d_eul_force_iter(nullptr),
                   num_points(0), num_bodies(0), iter_count(0) {}

    __host__
    void init_and_dispatch(std::vector<IBMBody> bodies) {
        num_points = 0;
        for (auto& b : bodies) {
            h_bodies.push_back(b);
            num_points += b.num_points;
            printf("adding body\n");
        }
        num_bodies = bodies.size();
        send_to_gpu();
    }

    __host__
    void send_to_gpu() {
        if (h_bodies.empty()) {
            printf("No bodies to send to GPU\n");
            return;
        }

        checkCudaErrors(cudaMalloc(&d_lag_points, dim * num_points * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_lag_u, dim * num_points * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_lag_rho, num_points * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_lag_force, dim * num_points * sizeof(float))); 
        
        checkCudaErrors(cudaMalloc(&d_eul_force_acc, dim * NX * NY * NZ * sizeof(float)));

        checkCudaErrors(cudaMalloc(&d_eul_u_prev, dim * NX * NY * NZ * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_eul_force_iter, dim * NX * NY * NZ * sizeof(float)));

        int offset = 0;
        for (const auto& body : h_bodies) {
            int num = body.num_points;
            checkCudaErrors(cudaMemcpy(d_lag_points + dim * offset, body.points, 
                            dim * num * sizeof(float), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_lag_u + dim * offset, body.velocities, 
                            dim * num * sizeof(float), cudaMemcpyHostToDevice));
            offset += num;
        }
        
        checkCudaErrors(cudaMemset(d_lag_force, 0.0f, dim * num_points * sizeof(float)));
        checkCudaErrors(cudaMemset(d_eul_force_acc, 0.0f, dim * NX * NY * NZ * sizeof(float)));

        printf("Transferred %d bodies with %d total points to GPU\n", num_bodies, num_points);
    }

    __host__
    void free() {
        if (d_lag_points != nullptr) {
            checkCudaErrors(cudaFree(d_lag_points));
            d_lag_points = nullptr;
        }
        if (d_lag_u != nullptr) {
            checkCudaErrors(cudaFree(d_lag_u));
            d_lag_u = nullptr;
        }
        if (d_lag_rho != nullptr) {
            checkCudaErrors(cudaFree(d_lag_rho));
            d_lag_rho = nullptr;
        }
        if (d_lag_force != nullptr) {
            checkCudaErrors(cudaFree(d_lag_force));
            d_lag_force = nullptr;
        }
        
        if (d_eul_force_acc != nullptr) {
            checkCudaErrors(cudaFree(d_eul_force_acc));
            d_eul_force_acc = nullptr;
        }
        
        if (d_eul_u_prev != nullptr) {
            checkCudaErrors(cudaFree(d_eul_u_prev));
            d_eul_u_prev = nullptr;
        }
        if (d_eul_force_iter != nullptr) {
            checkCudaErrors(cudaFree(d_eul_force_iter));
            d_eul_force_iter = nullptr;
        }
        
        for (auto& body : h_bodies) {
            h_ibm_free(body);
        }
        h_bodies.clear();
        num_points = 0;
        num_bodies = 0;
        printf("[IBM_MANAGER] Freeing\n");
    }

    ~IBMManager() {
        free();
    }

    void interpolate_velocity(float* d_eul_u, float* d_eul_rho) {
        dim3 threads(256);
        dim3 blocks(ceil(num_points / 256.0f) + 1);

        interpolate_velocities_kernel<dim><<<blocks, threads>>>(
            d_lag_points, d_lag_u, d_lag_rho, num_points, d_eul_u, d_eul_rho);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void spread_forces(float* d_eul_force_out) {
        dim3 threads(256);
        dim3 blocks(ceil(num_points / 256.0f) + 1);

        spread_forces_kernel<dim><<<blocks, threads>>>(
            d_lag_points, d_lag_force, num_points, d_eul_force_out);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void compute_lagrangian_correction(float* d_eul_rho) {
        dim3 threads(256);
        dim3 blocks(ceil(num_points / 256.0f) + 1);

        compute_lagrangian_kernel<dim><<<blocks, threads>>>(
            d_lag_points, d_lag_u, d_lag_force, num_points, d_lag_rho);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void correct_velocities(float* d_eul_u, float* d_eul_force_iter, float* d_eul_rho) {
        dim3 blocks, threads;
        if constexpr (dim == 2) {
            blocks = dim3((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                          (NY + BLOCK_SIZE - 1) / BLOCK_SIZE);
            threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
        }
        else {
            threads = dim3(8, 8, 4);
            blocks = dim3(((threads.x + NX - 1) / threads.x),
                          ((threads.y + NY - 1) / threads.y),
                          ((threads.z + NZ - 1) / threads.z));
        }

        correct_velocities_kernel<dim><<<blocks, threads>>>(
            d_eul_u, d_eul_force_iter, d_eul_rho);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void accumulate_forces(float* d_eul_force_total, float* d_eul_force_iter) {
        dim3 blocks, threads;
        if constexpr (dim == 2) {
            blocks = dim3((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                          (NY + BLOCK_SIZE - 1) / BLOCK_SIZE);
            threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
        }
        else {
            threads = dim3(8, 8, 4);
            blocks = dim3(((threads.x + NX - 1) / threads.x),
                          ((threads.y + NY - 1) / threads.y),
                          ((threads.z + NZ - 1) / threads.z));
        }

        accumulate_forces_kernel<dim><<<blocks, threads>>>(
            d_eul_force_total, d_eul_force_iter);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void multi_direct(float* d_rho_lbm, float* d_u_lbm, float* d_force_lbm) {
        iter_count = 0;
        if (num_bodies == 0)
            return;
        
        checkCudaErrors(cudaMemcpy(d_eul_u_prev, d_u_lbm, dim * NX * NY * NZ * sizeof(float), 
                                  cudaMemcpyDeviceToDevice));
        
        checkCudaErrors(cudaMemset(d_eul_force_acc, 0, dim * NX * NY * NZ * sizeof(float)));
        
        // printf("\n\n\n STARTING IBM ITER \n\n\n");
        for (int iter = 0; iter < ITER_MAX; iter++) {
            // printf("\n\nITER %d\n\n", iter);
            checkCudaErrors(cudaMemset(d_eul_force_iter, 0, dim * NX * NY * NZ * sizeof(float)));
            
            interpolate_velocity(d_eul_u_prev, d_rho_lbm);
            compute_lagrangian_correction(d_lag_rho);
            spread_forces(d_eul_force_iter);
            correct_velocities(d_eul_u_prev, d_eul_force_iter, d_rho_lbm);
            accumulate_forces(d_force_lbm, d_eul_force_iter);
            iter_count++;
        }
        
        // checkCudaErrors(cudaMemcpy(d_force_lbm, d_eul_force_acc, 
        //                           2 * NX * NY * sizeof(float), cudaMemcpyDeviceToDevice));
        // sum_arrays(d_force_lbm, d_eul_force_acc, d_force_lbm, 2*NX*NY);

        // checkCudaErrors(cudaMemcpy(d_u_lbm, d_eul_u_prev, 
        //                           2 * NX * NY * sizeof(float), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaDeviceSynchronize());
    }
};

#endif // ! IBMMANAGER_H