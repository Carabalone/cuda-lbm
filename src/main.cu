#include <stdio.h>
#include "core/lbm.cuh"
#include "functors/includes.cuh"
#include <iostream>
#include <chrono> 
#include "util/timer.cuh"
#include "IBM/IBMBody.cuh"

#if defined(USE_TAYLOR_GREEN)
    #include "scenarios/taylorGreen/TaylorGreenScenario.cuh"
    using Scenario = TaylorGreenScenario;
#elif defined(USE_POISEUILLE)
    #include "scenarios/poiseuille/PoiseuilleScenario.cuh"
    using Scenario = PoiseuilleScenario;
#elif defined(USE_LID_DRIVEN)
    #include "scenarios/lidDrivenCavity/lidDrivenCavityScenario.cuh"
    using Scenario = LidDrivenScenario;
#elif defined(USE_TURBULENT_CHANNEL)
    #include "scenarios/turbulentChannel/turbulentChannelScenario.cuh"
    using Scenario = TurbulentChannelScenario;
#elif defined(USE_FLOW_PAST_CYLINDER)
    #include "scenarios/flowPastCylinder/flowPastCylinderScenario.cuh"
    using Scenario = FlowPastCylinderScenario;
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


    constexpr float Re = compute_reynolds(Scenario::u_max, NY, Scenario::viscosity);
    std::cout << "Running " << Scenario::name() << " scenario" << std::endl;
    std::cout << "Viscosity: " << Scenario::viscosity 
              << ", Tau: " << Scenario::tau << std::endl;
    std::cout << "Reynolds number: " << Re << std::endl;

    LBM<dimensions> lbm; // idea is control from host and give args to the kernels for the device.

    // destructor frees automatically
    lbm.allocate<Scenario>();

    const int total_timesteps = 30000;
    const int save_int = 100;
    int t = 0;

    lbm.init<Scenario>();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    Timer simulation_timer;
    float last_progress_time = 0.0f;
    simulation_timer.reset();

    while (t < total_timesteps) {
        // printf("\n----------------------------------------NEW_TS{%d}----------------------------------------\n",t);
        bool save = (t+1)%save_int == 0;
        cudaEventRecord(start);

        lbm.increase_ts<Scenario>();

        lbm.stream();
        lbm.swap_buffers();

        lbm.apply_boundaries<Scenario>();
        
        lbm.uncorrected_macroscopics();

        // -----------IBM stuff-----------------
        // lbm.reset_forces<Scenario>();
        // lbm.ibm_step();
        // -------------------------------------

        lbm.correct_macroscopics();

        lbm.compute_equilibrium();
        lbm.collide<Scenario::CollisionOp>();
        

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        float elapsed = simulation_timer.elapsed_seconds();
        simulation_timer.checkpoint_seconds();

        if (t % (total_timesteps / 20) == 0) {
            last_progress_time = elapsed;
            float progress = (t * 100.0f) / total_timesteps;
            printf("Simulation progress: %.1f%% (timestep %d/%d)\n", progress, t, total_timesteps);
            float median_ts = simulation_timer.median_timestep();
            float remaining = median_ts * (total_timesteps - t);
            
            std::cout << std::fixed << std::setprecision(1);
            std::cout << "Median timestep: " << median_ts * 1000.0f << " ms | ";
            
            int rem_hours = static_cast<int>(remaining / 3600);
            int rem_mins  = static_cast<int>((remaining - rem_hours * 3600) / 60);
            int rem_secs  = static_cast<int>(remaining - rem_hours * 3600 - rem_mins * 60);
            
            std::cout << "Est. remaining: ";
            if (rem_hours > 0) std::cout << rem_hours << "h ";
            std::cout << rem_mins << "m " << rem_secs << "s" << std::endl;
        }
        if (save) {
            // lbm.save_macroscopics(t+1); // save macroscopics updates the data from GPU to CPU.
            if constexpr (Scenario::has_analytical_solution) {
                // auto start = std::chrono::high_resolution_clock::now();

                lbm.update_macroscopics();
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

    // destructor frees automatically

    return 0;
}
