#include <stdio.h>
#include "lbm.cuh"
#include <iostream>


void setup_cuda() {
    checkCudaErrors(cudaSetDevice(0));
    int deviceId = 0;
    checkCudaErrors(cudaGetDevice(&deviceId));

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, deviceId));

    size_t gpu_free_mem, gpu_total_mem;
    checkCudaErrors(cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem));

    printf("CUDA information\n");
    printf("       using device: %d\n", deviceId);
    printf("               name: %s\n",deviceProp.name);
    printf("    multiprocessors: %d\n", deviceProp.multiProcessorCount);
    printf(" compute capability: %d.%d\n", deviceProp.major,deviceProp.minor);
    printf("      global memory: %.1f MiB\n", deviceProp.totalGlobalMem/1048576.0);
    printf("        free memory: %.1f MiB\n", gpu_free_mem/1048576.0);
    printf("\n");

}

int main(void) {
    setup_cuda();

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    std::cout << viscosity_to_tau(0.8) << std::endl;

    LBM lbm; // idea is control from host and give args to the kernels for the device.
    lbm.allocate();
    
    // while (timestaps) {
    //     lbm.stream(); 
    //     lbm.update_macroscopic();
    //     lbm.calc_equilibrium <
    //     lbm.collide();
    //     lbm.process_boundary();
    // }

    lbm.init();
    lbm.macroscopics();

    lbm.free();

    return 0;
}
