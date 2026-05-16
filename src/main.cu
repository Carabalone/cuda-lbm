#include <stdio.h>
#include "core/lbm.cuh"
#include "functors/includes.cuh"
#include <cstdlib>
#include <iostream>
#include <chrono> 
#include <iomanip>
#include <algorithm>
#include <string>
#include <type_traits>
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
#elif defined(USE_TAYLOR_GREEN_3D)
    #include "scenarios/taylorGreen3D/taylorGreen3DScenario.cuh"
    using Scenario = TaylorGreen3DScenario;
#elif defined(USE_FLOW_PAST_SQUARE_CYLINDER)
    #include "scenarios/flowPastSquareCylinder/flowPastSquareCylinderScenario.cuh"
    using Scenario = FlowPastSquareCylinderScenario;
#elif defined(USE_ZERO_FLOW_TEST)
    #include "scenarios/zeroFlow/zeroFlowScenario.cuh"
    using Scenario = ZeroFlowScenario;
#elif defined(USE_WIND_TUNNEL)
    #include "scenarios/windTest/windTestScenario.cuh"
    using Scenario = WindTunnelScenario;
#endif

template <typename CollisionType>
struct CollisionNeedsEquilibrium : std::true_type {};

template <int dim>
struct CollisionNeedsEquilibrium<CM<dim, NoAdapter>> : std::false_type {};

// TGV fast path (periodic, no forcing): always on when this scenario is selected.
#if defined(USE_TAYLOR_GREEN_3D)
constexpr bool kFusedStreamMacros = true;
constexpr bool kSkipBoundaries = true;
constexpr bool kSkipForceKernels = true;
constexpr bool kAsyncKernels = true;
#else
constexpr bool kFusedStreamMacros = false;
constexpr bool kSkipBoundaries = false;
constexpr bool kSkipForceKernels = false;
constexpr bool kAsyncKernels = false;
#endif

void setup_cuda(bool verbose) {
    checkCudaErrors(cudaSetDevice(0));
    int deviceId = 0;
    checkCudaErrors(cudaGetDevice(&deviceId));

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, deviceId));

    size_t gpu_free_mem, gpu_total_mem;
    checkCudaErrors(cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem));

    if (!verbose) {
        return;
    }
    printf("CUDA information\n");
    printf("       using device: %d\n", deviceId);
    printf("               name: %s\n",deviceProp.name);
    printf("    multiprocessors: %d\n", deviceProp.multiProcessorCount);
    printf(" compute capability: %d.%d\n", deviceProp.major,deviceProp.minor);
    printf("      global memory: %.1f MiB\n", deviceProp.totalGlobalMem/1048576.0);
    printf("        free memory: %.1f MiB\n", gpu_free_mem/1048576.0);
    printf("\n");

}

struct RuntimeOptions {
    bool benchmark_mode = false;
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "Options:\n"
        << "  --benchmark                 Run in benchmark mode (scenario-controlled settings)\n"
        << "  --help                      Show this help message\n";
}

bool parse_runtime_options(int argc, char** argv, RuntimeOptions* options) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }

        if (arg == "--benchmark") {
            options->benchmark_mode = true;
            continue;
        }

        std::cerr << "Unknown option: " << arg << std::endl;
        print_usage(argv[0]);
        return false;
    }

    return true;
}

int main(int argc, char** argv) {
    RuntimeOptions options;
    if (!parse_runtime_options(argc, argv, &options)) {
        return 1;
    }

    const bool benchmark = options.benchmark_mode;
    const int total_timesteps = benchmark ? Scenario::benchmark_timesteps : Scenario::timesteps;
    const int benchmark_warmup = benchmark ? Scenario::benchmark_warmup : 0;
    const int save_int = benchmark ? 0 : Scenario::save_interval;

    const bool disable_vtk = benchmark;
    const bool disable_validation = benchmark;
    const bool disable_ibm = benchmark;
    const bool async_kernels = kAsyncKernels || benchmark;

    setup_cuda(!benchmark);


    constexpr float Re = compute_reynolds(Scenario::u_max, NY, Scenario::viscosity);
    std::cout << "Running " << Scenario::name() << " scenario" << std::endl;
    std::cout << "Viscosity: " << Scenario::viscosity 
              << ", Tau: " << Scenario::tau << std::endl;
    std::cout << "Reynolds number: " << Re << std::endl;
    std::cout << "[Timestep] Fused stream+macros: " << (kFusedStreamMacros ? "yes" : "no") << std::endl;
    std::cout << "[Timestep] Skip boundaries: " << (kSkipBoundaries ? "yes" : "no") << std::endl;
    std::cout << "[Timestep] Skip force kernels: " << (kSkipForceKernels ? "yes" : "no") << std::endl;
    std::cout << "[Timestep] Async kernels: " << (async_kernels ? "yes" : "no") << std::endl;
    if (benchmark) {
        std::cout << "[Benchmark] Enabled" << std::endl;
        std::cout << "[Benchmark] Timesteps: " << total_timesteps
                  << ", Warmup: " << benchmark_warmup << std::endl;
        std::cout << "[Benchmark] IBM disabled: " << (disable_ibm ? "yes" : "no") << std::endl;
        std::cout << "[Benchmark] Async kernels: " << (async_kernels ? "yes" : "no") << std::endl;
    }

    LBM<dimensions> lbm; // idea is control from host and give args to the kernels for the device.

    // destructor frees automatically
    lbm.allocate<Scenario>();
    lbm.set_kernel_synchronization(!async_kernels);
    lbm.set_collect_moment_stats(!benchmark && !kFusedStreamMacros);

    int t = 0;

    lbm.init<Scenario>();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEvent_t benchmark_start, benchmark_stop;
    cudaEventCreate(&benchmark_start);
    cudaEventCreate(&benchmark_stop);
    Timer simulation_timer;
    simulation_timer.reset();
    double measured_ms_total = 0.0;
    int measured_steps = 0;
    const double lattice_nodes = static_cast<double>(NX) * static_cast<double>(NY) * static_cast<double>(NZ);

    while (t < total_timesteps) {
        // printf("\n----------------------------------------NEW_TS{%d}----------------------------------------\r",t);
        bool save = save_int > 0 && ((t + 1) % save_int == 0);
        if (!benchmark) {
            cudaEventRecord(start);
        } else if (t == std::min(benchmark_warmup, total_timesteps)) {
            cudaEventRecord(benchmark_start);
        }

        lbm.increase_ts<Scenario>();

        if (kFusedStreamMacros) {
            lbm.stream_and_uncorrected_macroscopics();
            lbm.swap_buffers();
        } else {
            lbm.stream();
            lbm.swap_buffers();
        }

        if (!kSkipBoundaries) {
            lbm.apply_boundaries<Scenario>();
        }

        if (!kFusedStreamMacros) {
            lbm.uncorrected_macroscopics();
        }

        // -----------IBM stuff-----------------
        if (!kSkipForceKernels) {
            lbm.reset_forces<Scenario>();
            if (!disable_ibm) {
                lbm.ibm_step();
            }
            lbm.correct_macroscopics();
        } else if (!disable_ibm) {
            lbm.ibm_step();
        }
        // -------------------------------------

        if constexpr (CollisionNeedsEquilibrium<typename Scenario::CollisionOp>::value) {
            lbm.compute_equilibrium();
        }
        lbm.collide<Scenario::CollisionOp>();
        

        if (!benchmark) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
        }

        if (!benchmark) {
            simulation_timer.checkpoint_seconds();
        }

        if (!benchmark && t % 100 == 0) {
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
            // lbm.save_midplane_slice(t+1);
            if (!disable_vtk) {
                lbm.save_vtk(t+1);
            }
            if (!disable_validation) {
                if constexpr (Scenario::has_analytical_solution) {
                    lbm.update_macroscopics();
                    // printf("Dimensionless Time: %f\n", (t / ((NX / (2*M_PI)) / 0.05f)));
                    lbm.compute_error<Scenario>();
                }
            }
        }

        t++;
    }

    if (benchmark) {
        const int warmup = std::min(benchmark_warmup, total_timesteps);
        if (total_timesteps > warmup) {
            cudaEventRecord(benchmark_stop);
            cudaEventSynchronize(benchmark_stop);

            float measured_ms = 0.0f;
            cudaEventElapsedTime(&measured_ms, benchmark_start, benchmark_stop);
            measured_ms_total = measured_ms;
            measured_steps = total_timesteps - warmup;
        }

        std::cout << "\n[Benchmark] Summary\n";
        if (measured_steps == 0) {
            std::cout << "No measured steps were collected." << std::endl;
        } else {
            double avg_ms = measured_ms_total / static_cast<double>(measured_steps);
            double steps_per_sec = 1000.0 / avg_ms;
            double mlups = (lattice_nodes * steps_per_sec) / 1.0e6;
            std::cout << "Measured steps: " << measured_steps << std::endl;
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "Average timestep: " << avg_ms << " ms" << std::endl;
            std::cout << "Throughput: " << steps_per_sec << " steps/s" << std::endl;
            std::cout << "Performance: " << mlups << " MLUPS" << std::endl;
        }
    }

    // destructor frees automatically

    return 0;
}
