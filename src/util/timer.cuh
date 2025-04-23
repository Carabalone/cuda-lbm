#include <stdio.h>
#include "core/lbm.cuh"
#include "functors/includes.cuh"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <algorithm>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point last_checkpoint;
    std::vector<float> timestep_durations;
    
public:
    Timer() {
        reset();
    }
    
    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
        last_checkpoint = start_time;
        timestep_durations.clear();
    }
    
    float elapsed_seconds() {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = now - start_time;
        return elapsed.count();
    }
    
    float checkpoint_seconds() {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = now - last_checkpoint;
        last_checkpoint = now;
        
        timestep_durations.push_back(elapsed.count());
        if (timestep_durations.size() > 100) {
            timestep_durations.erase(timestep_durations.begin());
        }
        
        return elapsed.count();
    }
    
    float median_timestep() {
        if (timestep_durations.empty()) return 0.0f;
        
        std::vector<float> sorted = timestep_durations;
        std::sort(sorted.begin(), sorted.end());
        
        if (sorted.size() % 2 == 0) {
            return (sorted[sorted.size()/2 - 1] + sorted[sorted.size()/2]) / 2.0f;
        } else {
            return sorted[sorted.size()/2];
        }
    }
};

float calculate_memory_bandwidth(int num_nodes, int num_populations, float elapsed_seconds) {
    // Each node reads/writes multiple float values per iteration:
    // - Distributions (f and f_back): 2 * num_populations * sizeof(float)
    // - Equilibrium: num_populations * sizeof(float)
    // - Density: 1 * sizeof(float)
    // - Velocity: 2 * sizeof(float) for 2D
    // - Force: 2 * sizeof(float) for 2D
    
    size_t bytes_per_iter = num_nodes * (
        2 * num_populations * sizeof(float) +  // f and f_back
        num_populations * sizeof(float) +      // f_eq
        sizeof(float) +                        // rho
        2 * sizeof(float) +                    // u (2D)
        2 * sizeof(float)                      // force (2D)
    );
    
    return (bytes_per_iter / (1024.0f * 1024.0f * 1024.0f)) / elapsed_seconds;  // GB/s
}