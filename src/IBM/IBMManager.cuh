#ifndef IBMMANAGER_H
#define IBMMANAGER_H
#include "IBM/IBMBody.cuh"

#include <vector>

struct IBMManager {

    std::vector<IBMBody> h_bodies;
    IBMBody **d_bodies;
    int num_bodies;

    IBMManager() : d_bodies(nullptr), num_bodies(0) {}

    __host__
    void init_and_dispatch(std::vector<IBMBody> bodies) {
        for (auto& b : bodies) {
            h_bodies.push_back(b);
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

        checkCudaErrors(cudaMalloc((void**) &d_bodies, num_bodies * sizeof(IBMBody*)));

        for (int i=0; i < num_bodies; i++) {
            IBMBody* d_ptr = IBMBody::copy_to_gpu(h_bodies[i]);
            checkCudaErrors(cudaMemcpy(&d_bodies[i], &d_ptr, sizeof(IBMBody*), cudaMemcpyHostToDevice));
        }
        printf("Transferred %d bodies to GPU\n", num_bodies);
    }

    __host__
    void free() {
        if (d_bodies == nullptr) return;
        
        IBMBody** h_ptrs = new IBMBody*[num_bodies];
        checkCudaErrors(cudaMemcpy(h_ptrs, d_bodies, 
                       num_bodies * sizeof(IBMBody*), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < num_bodies; i++) {
            if (h_ptrs[i] != nullptr) {
                IBMBody::gpu_free(h_ptrs[i]);
            }
        }
        
        checkCudaErrors(cudaFree(d_bodies));
        d_bodies = nullptr;
        
        delete[] h_ptrs;

        for (auto& body : h_bodies) {
            h_ibm_free(body);
        }
        h_bodies.clear();
        num_bodies = 0;

        printf("[IBM_MANAGER] Freeing");
    }

    ~IBMManager() {
        free();
    }
};

#endif // ! IBMMANAGER_H