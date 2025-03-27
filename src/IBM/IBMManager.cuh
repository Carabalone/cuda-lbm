#ifndef IBMMANAGER_H
#define IBMMANAGER_H
#include "IBM/IBMBody.cuh"

#include <vector>

__global__
void interpolate_velocities_kernel(float* points, float* u_ibm, int num_points, float* u);

__global__
void compute_penalties_kernel(float* u_ibm, float* penalties, int num_points, float* rho);

__global__
void spread_forces_kernel(float* points, float* penalties, int num_points, float* lbm_forces);

struct IBMManager {

    std::vector<IBMBody> h_bodies;
    float* d_points;
    float* d_velocities;
    float* d_penalties;
    int num_points;
    int num_bodies;

    IBMManager() : d_points(nullptr), d_velocities(nullptr), num_points(0), num_bodies(0),
                   d_penalties(nullptr) {}

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

        checkCudaErrors(cudaMalloc(&d_points, 2 * num_points * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_velocities, 2 * num_points * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_penalties, 2 * num_points * sizeof(float)));

        int offset = 0;
        for (const auto& body : h_bodies) {
            int num = body.num_points;
            checkCudaErrors(cudaMemcpy(d_points + 2 * offset, body.points, 
                            2 * num * sizeof(float), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_velocities + 2 * offset, body.velocities, 
                            2 * num * sizeof(float), cudaMemcpyHostToDevice));
                            offset += num;
        }
        checkCudaErrors(cudaMemset(d_penalties, 0.0f, 2 * num_points * sizeof(float)));

        printf("Transferred %d bodies with %d total points to GPU\n", num_bodies, num_points);
    }

    __host__
    void free() {
        if (d_points != nullptr) {
            checkCudaErrors(cudaFree(d_points));
            d_points = nullptr;
        }
        if (d_velocities != nullptr) {
            checkCudaErrors(cudaFree(d_velocities));
            d_velocities = nullptr;
        }
        if (d_penalties != nullptr) {
            checkCudaErrors(cudaFree(d_penalties));
            d_penalties = nullptr;
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

    void interpolate(float* d_u) {
        dim3 threads(256);
        dim3 blocks(ceil(num_points / BLOCK_SIZE) + 1);

        interpolate_velocities_kernel<<<threads, blocks>>>(d_points, d_velocities, num_points, d_u);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void compute_penalties(float* d_rho) {
        dim3 threads(256);
        dim3 blocks(ceil(num_points / BLOCK_SIZE) + 1);

        compute_penalties_kernel<<<threads, blocks>>>(d_velocities, d_penalties, num_points, d_rho);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void spread_forces(float* d_forces) {
        dim3 threads(256);
        dim3 blocks(ceil(num_points / BLOCK_SIZE) + 1);

        spread_forces_kernel<<<threads, blocks>>>(d_points, d_penalties, num_points, d_forces);
        checkCudaErrors(cudaDeviceSynchronize());
    }
};

#endif // ! IBMMANAGER_H