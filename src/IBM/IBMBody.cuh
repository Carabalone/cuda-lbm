#ifndef IBM_BODY_H
#define IBM_BODY_H

#include <math.h>
#include "util/utility.cuh"

struct IBMBody {
    int num_points;
    float* points;
    float* velocities;

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