#ifndef IBMMANAGER_H
#define IBMMANAGER_H
#include "IBM/IBMBody.cuh"

#include <vector>

struct IBMManager {

    std::vector<IBMBody> h_bodies;
    IBMBody **d_bodies;
    float* d_points;
    float* d_velocities;
    int num_points;
    int num_bodies;

    IBMManager() : d_points(nullptr), d_velocities(nullptr), num_points(0), num_bodies(0) {}

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

        int offset = 0;
        for (const auto& body : h_bodies) {
            int num = body.num_points;
            checkCudaErrors(cudaMemcpy(d_points + 2 * offset, body.points, 
                            2 * num * sizeof(float), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_velocities + 2 * offset, body.velocities, 
                            2 * num * sizeof(float), cudaMemcpyHostToDevice));
            offset += num;
        }

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
};

#endif // ! IBMMANAGER_H