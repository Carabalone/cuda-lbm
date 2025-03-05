#include <stdio.h>
#include "lbm.cuh"
#include "functors/includes.cuh"
#include <iostream>

#define SCENARIO TAYLOR_GREEN

#if defined(SCENARIO) && SCENARIO == TAYLOR_GREEN
    #include "scenarios/taylorGreen2D/TaylorGreenScenario.cuh"
    using Scenario = TaylorGreenScenario;
#else
    #include "scenarios/poiseuille/PoiseuilleScenario.cuh"
    using Scenario = PoiseuilleScenario;
#endif


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

    std::cout << "Running " << Scenario::name() << " scenario" << std::endl;
    std::cout << "Viscosity: " << Scenario::viscosity 
              << ", Tau: " << Scenario::tau << std::endl;

    LBM lbm; // idea is control from host and give args to the kernels for the device.
    lbm.allocate();

    const int total_timesteps = 10000;
    const int save_int = 100;
    int t = 0;

    lbm.init<Scenario>();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    while (t < total_timesteps) {
        bool save = (t+1)%save_int == 0;
        cudaEventRecord(start);

        lbm.increase_ts<Scenario>();

        lbm.stream();
        lbm.swap_buffers();

        lbm.apply_boundaries();

        lbm.macroscopics();
        lbm.compute_equilibrium();
        lbm.collide();


        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        if (t % (total_timesteps / 20) == 0) {
            float progress = (t * 100.0f) / total_timesteps;
            printf("Simulation progress: %.1f%% (timestep %d/%d)\n", progress, t, total_timesteps);
        }
        if (save) {
            lbm.save_macroscopics(t+1); // save macroscopics updates the data from GPU to CPU.
            if constexpr (Scenario::has_analytical_solution) {
                printf("%s[%d]: error, %.2f%%\n", 
                       Scenario::name(), t+1,
                       lbm.compute_error<Scenario>());
            }
        }

        t++;
    }

    lbm.free();

    return 0;
}
