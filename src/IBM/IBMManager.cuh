#ifndef IBMMANAGER_H
#define IBMMANAGER_H
#include "IBM/IBMBody.cuh"
#include "IBM/IBMUtils.cuh"

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

// __device__ int num_ibm_points;

template <int dim>
struct IBMManager {
    float* d_lag_points;        // Boundary point coordinates [2*num_points]
    float* d_lag_u;             // Velocities at boundary points [2*num_points]
    float* d_lag_rho;           // Densities at boundary points [num_points]
    float* d_lag_force;         // Forces at boundary points [2*num_points]
    
    float* d_eul_force_acc;     // Accumulated forces on grid [2*NX*NY]
    
    float* d_eul_u_prev;        // Previous iteration velocities [2*NX*NY]
    float* d_eul_force_iter;    // Current iteration forces [2*NX*NY]
    
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

        // NZ defaults as 1 if D2Q9
        checkCudaErrors(cudaMalloc(&d_lag_points, dim * num_points * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_lag_u, dim * num_points * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_lag_rho, num_points * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_lag_force, dim * num_points * sizeof(float))); 
        
        checkCudaErrors(cudaMalloc(&d_eul_force_acc, dim * NX * NY * NZ * sizeof(float)));

        checkCudaErrors(cudaMalloc(&d_eul_u_prev, dim * NX * NY * NZ * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_eul_force_iter, dim * NX * NY * NZ * sizeof(float)));

        #ifdef IBM_SOA
            printf("\n\nSOA\n\n");
            int offset = 0;
            for (const auto& body : h_bodies) {
                int num_pts = body.num_points;
                checkCudaErrors(cudaMemcpy(d_lag_points + dim * offset, body.points, 
                                dim * num_pts * sizeof(float), cudaMemcpyHostToDevice));
                offset += num_pts;
            }

            float* h_soa_velocities = new float[dim * this->num_points];

            printf("[SENDING] IBM points: %d\n", this->num_points);
            int curr_point_offset = 0;
            for (const auto& body : h_bodies) {
                // for now points stay in AoS (will make morton code optimizatino later.)

                for (int body_point = 0; body_point < body.num_points; body_point++) {
                    for (int component = 0; component < dim; component++) {
                        h_soa_velocities[component * this->num_points + curr_point_offset] =
                            body.velocities[body_point * dim + component];
                    }
                    curr_point_offset++;
                }
            }

            checkCudaErrors(cudaMemcpy(d_lag_u,
                                       h_soa_velocities,
                                       dim * this->num_points * sizeof(float),
                                       cudaMemcpyHostToDevice));

            delete[] h_soa_velocities;
        #elif defined(IBM_CSOA)
            LBM_ASSERT(false, "SUPPORT FOR CSOA NOT AVAILABLE YET");
        #else
            printf("\n\nNORMAL\n\n");
            int offset = 0;
            for (const auto& body : h_bodies) {
                int num_pts = body.num_points;
                checkCudaErrors(cudaMemcpy(d_lag_points + dim * offset, body.points, 
                                dim * num_pts * sizeof(float), cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemcpy(d_lag_u + dim * offset, body.velocities, 
                                dim * num_pts * sizeof(float), cudaMemcpyHostToDevice));
                offset += num_pts;
            }
        #endif
        
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
        dim3 blocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (NY+BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

        correct_velocities_kernel<dim><<<blocks, threads>>>(
            d_eul_u, d_eul_force_iter, d_eul_rho);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void accumulate_forces(float* d_eul_force_total, float* d_eul_force_iter) {
        dim3 blocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (NY+BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

        accumulate_forces_kernel<dim><<<blocks, threads>>>(
            d_eul_force_total, d_eul_force_iter);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void multi_direct(float* d_rho_lbm, float* d_u_lbm, float* d_force_lbm) {
        iter_count = 0;
        if (num_bodies == 0)
            return;
        
        // NZ defaults as 1 if D2Q9
        checkCudaErrors(cudaMemcpy(d_eul_u_prev, d_u_lbm, dim * NX * NY * NZ * sizeof(float), 
                                  cudaMemcpyDeviceToDevice));
        
        checkCudaErrors(cudaMemset(d_eul_force_acc, 0, dim * NX * NY * NZ * sizeof(float)));
        
        for (int iter = 0; iter < ITER_MAX; iter++) {
            checkCudaErrors(cudaMemset(d_eul_force_iter, 0, dim * NX * NY  * NZ * sizeof(float)));
            
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

#include "IBM_impl.cuh"

#endif // ! IBMMANAGER_H