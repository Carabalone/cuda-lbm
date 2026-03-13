#include <stdio.h>
#include "core/lbm.cuh"
#include "functors/includes.cuh"
#include <cstdlib>
#include <iostream>
#include <chrono> 
#include <iomanip>
#include <limits>
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

struct RuntimeOptions {
    int total_timesteps = 8000;
    int save_int = 100;
    bool benchmark_mode = false;
    int benchmark_warmup = 50;
    bool disable_vtk = false;
    bool disable_validation = false;
    bool disable_ibm = false;
    bool async_kernels = false;
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "Options:\n"
        << "  --benchmark                 Run in benchmark mode (no VTK/validation output)\n"
        << "  --timesteps <N>             Number of timesteps to run (default: 8000)\n"
        << "  --save-int <N>              Save interval for VTK/validation (0 disables saves, default: 100)\n"
        << "  --benchmark-warmup <N>      Warmup steps excluded from benchmark metrics (default: 50)\n"
        << "  --no-vtk                    Disable VTK output\n"
        << "  --no-validation             Disable analytical validation output\n"
        << "  --no-ibm                    Disable IBM step in the timestep loop\n"
        << "  --async-kernels             Avoid per-kernel cudaDeviceSynchronize in hot path\n"
        << "  --help                      Show this help message\n";
}

bool parse_non_negative_int(const char* text, int* out) {
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 || value > std::numeric_limits<int>::max()) {
        return false;
    }
    *out = static_cast<int>(value);
    return true;
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
        if (arg == "--no-vtk") {
            options->disable_vtk = true;
            continue;
        }
        if (arg == "--no-validation") {
            options->disable_validation = true;
            continue;
        }
        if (arg == "--no-ibm") {
            options->disable_ibm = true;
            continue;
        }
        if (arg == "--async-kernels") {
            options->async_kernels = true;
            continue;
        }
        if (arg == "--timesteps" || arg == "--save-int" || arg == "--benchmark-warmup") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for option: " << arg << std::endl;
                return false;
            }
            int value = 0;
            if (!parse_non_negative_int(argv[++i], &value)) {
                std::cerr << "Invalid numeric value for option " << arg << ": " << argv[i] << std::endl;
                return false;
            }

            if (arg == "--timesteps") {
                if (value <= 0) {
                    std::cerr << "--timesteps must be > 0" << std::endl;
                    return false;
                }
                options->total_timesteps = value;
            } else if (arg == "--save-int") {
                options->save_int = value;
            } else {
                options->benchmark_warmup = value;
            }
            continue;
        }

        std::cerr << "Unknown option: " << arg << std::endl;
        print_usage(argv[0]);
        return false;
    }

    if (options->benchmark_mode) {
        options->disable_vtk = true;
        options->disable_validation = true;
        options->disable_ibm = true;
        options->async_kernels = true;
        options->save_int = 0;
        if (options->benchmark_warmup >= options->total_timesteps) {
            std::cerr << "[Benchmark] Warmup >= total timesteps, clamping warmup to 0." << std::endl;
            options->benchmark_warmup = 0;
        }
    }

    return true;
}

int main(int argc, char** argv) {
    RuntimeOptions options;
    if (!parse_runtime_options(argc, argv, &options)) {
        return 1;
    }

    printf("started");
    setup_cuda();


    constexpr float Re = compute_reynolds(Scenario::u_max, NY, Scenario::viscosity);
    std::cout << "Running " << Scenario::name() << " scenario" << std::endl;
    std::cout << "Viscosity: " << Scenario::viscosity 
              << ", Tau: " << Scenario::tau << std::endl;
    std::cout << "Reynolds number: " << Re << std::endl;
    if (options.benchmark_mode) {
        std::cout << "[Benchmark] Enabled" << std::endl;
        std::cout << "[Benchmark] Timesteps: " << options.total_timesteps
                  << ", Warmup: " << options.benchmark_warmup << std::endl;
        std::cout << "[Benchmark] IBM disabled: " << (options.disable_ibm ? "yes" : "no") << std::endl;
        std::cout << "[Benchmark] Async kernels: " << (options.async_kernels ? "yes" : "no") << std::endl;
    }

    LBM<dimensions> lbm; // idea is control from host and give args to the kernels for the device.

    // destructor frees automatically
    lbm.allocate<Scenario>();
    lbm.set_kernel_synchronization(!options.async_kernels);
    lbm.set_collect_moment_stats(!options.benchmark_mode);

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

    while (t < options.total_timesteps) {
        // printf("\n----------------------------------------NEW_TS{%d}----------------------------------------\r",t);
        bool save = options.save_int > 0 && ((t + 1) % options.save_int == 0);
        if (!options.benchmark_mode) {
            cudaEventRecord(start);
        } else if (t == options.benchmark_warmup) {
            cudaEventRecord(benchmark_start);
        }

        lbm.increase_ts<Scenario>();

        lbm.stream();
        lbm.swap_buffers();

        lbm.apply_boundaries<Scenario>();

        lbm.uncorrected_macroscopics();

        // -----------IBM stuff-----------------
        lbm.reset_forces<Scenario>();
        if (!options.disable_ibm) {
            lbm.ibm_step();
        }
        // -------------------------------------

        lbm.correct_macroscopics();

        if constexpr (CollisionNeedsEquilibrium<typename Scenario::CollisionOp>::value) {
            lbm.compute_equilibrium();
        }
        lbm.collide<Scenario::CollisionOp>();
        

        if (!options.benchmark_mode) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
        }

        if (!options.benchmark_mode) {
            simulation_timer.checkpoint_seconds();
        }

        if (!options.benchmark_mode && t % 100 == 0) {
            float progress = (t * 100.0f) / options.total_timesteps;
            printf("Simulation progress: %.1f%% (timestep %d/%d)\n", progress, t, options.total_timesteps);
            float median_ts = simulation_timer.median_timestep();
            float remaining = median_ts * (options.total_timesteps - t);
            
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
            if (!options.disable_vtk) {
                lbm.save_vtk(t+1);
            }
            if (!options.disable_validation) {
                if constexpr (Scenario::has_analytical_solution) {
                    lbm.update_macroscopics();
                    // printf("Dimensionless Time: %f\n", (t / ((NX / (2*M_PI)) / 0.05f)));
                    lbm.compute_error<Scenario>();
                }
            }
        }

        t++;
    }

    if (options.benchmark_mode) {
        if (options.total_timesteps > options.benchmark_warmup) {
            cudaEventRecord(benchmark_stop);
            cudaEventSynchronize(benchmark_stop);

            float measured_ms = 0.0f;
            cudaEventElapsedTime(&measured_ms, benchmark_start, benchmark_stop);
            measured_ms_total = measured_ms;
            measured_steps = options.total_timesteps - options.benchmark_warmup;
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
