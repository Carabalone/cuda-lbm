#ifndef IBM_BODY_H
#define IBM_BODY_H

#include <math.h>
#include "util/utility.cuh"
#include "core/lbm_constants.cuh"

struct IBMBody {
    int num_points;
    float* points;
    float* velocities;

    __host__
    static IBMBody* copy_to_gpu(IBMBody h_body) {
        IBMBody* d_body;

        checkCudaErrors(cudaMalloc((void**) &d_body, sizeof(IBMBody)));
        IBMBody temp = h_body;
        temp.points = nullptr;
        temp.velocities = nullptr;

        checkCudaErrors(cudaMemcpy(d_body, &temp, sizeof(IBMBody), cudaMemcpyHostToDevice));

        float* d_points;
        checkCudaErrors(cudaMalloc(&d_points, dimensions * h_body.num_points * sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_points, h_body.points, 
                      dimensions * h_body.num_points * sizeof(float), cudaMemcpyHostToDevice));

        float* d_velocities;
        checkCudaErrors(cudaMalloc(&d_velocities, dimensions * h_body.num_points * sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_velocities, h_body.velocities, 
                      dimensions * h_body.num_points * sizeof(float), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMemcpy(&(d_body->points), &d_points, sizeof(float*),
                        cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(&(d_body->velocities), &d_velocities, sizeof(float*),
                        cudaMemcpyHostToDevice));
        
        return d_body;

    }

    __host__
    static void gpu_free(IBMBody* d_body) {
        if (d_body == nullptr) return;
        
        // host copy to access the device pointers
        IBMBody h_body;
        checkCudaErrors(cudaMemcpy(&h_body, d_body, sizeof(IBMBody), cudaMemcpyDeviceToHost));
        
        if (h_body.points != nullptr) {
            checkCudaErrors(cudaFree(h_body.points));
        }
        
        if (h_body.velocities != nullptr) {
            checkCudaErrors(cudaFree(h_body.velocities));
        }
        
        checkCudaErrors(cudaFree(d_body));
    }
};

// assumes grid coordinates (system, but doesn't need to be aligned)
IBMBody create_cylinder(float cx, float cy, float r, int num_pts=16);

inline void h_ibm_free(IBMBody body) {
    if (body.points != nullptr) {
        delete[] body.points;
        body.points = nullptr;
    }
    
    if (body.velocities != nullptr) {
        delete[] body.velocities;
        body.velocities = nullptr;
    }
    
    body.num_points = 0;
}

#endif // ! IBM_BODY_H