#include <stdio.h>
#include "lbm.cuh"
#include "functors/includes.cuh"
#include <iostream>
#include <chrono> 

#if defined(USE_TAYLOR_GREEN)
    #include "scenarios/taylorGreen2D/TaylorGreenScenario.cuh"
    using Scenario = TaylorGreenScenario;
#elif defined(USE_POISEUILLE)
    #include "scenarios/poiseuille/PoiseuilleScenario.cuh"
    using Scenario = PoiseuilleScenario;
#elif defined(USE_LID_DRIVEN)
    #include "scenarios/lidDrivenCavity2D/lidDrivenCavityScenario.cuh"
    using Scenario = LidDrivenScenario;
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

    const int total_timesteps = 30000;
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

        lbm.apply_boundaries<Scenario>();
        
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
                // auto start = std::chrono::high_resolution_clock::now();

                printf("%s[%d]: error, %.2f%%\n", 
                       Scenario::name(), t+1,
                       lbm.compute_error<Scenario>());

                // auto end = std::chrono::high_resolution_clock::now();
                // std::chrono::duration<double> duration = end - start;
                // printf("Time taken to compute error: %.1f ms\n", duration.count() * 1000.0f);
            }
        }

        t++;
    }

    lbm.free();

    return 0;
}
