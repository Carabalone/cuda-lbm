#ifndef TURBULENT_CHANNEL_FUNCTORS_H
#define TURBULENT_CHANNEL_FUNCTORS_H

#include "core/lbm_constants.cuh"
#include "defines.hpp"

struct TurbulentChannelInit {
    float u_max;
    float perturbation;
    
    TurbulentChannelInit(float u_max, float perturbation) 
        : u_max(u_max), perturbation(perturbation) {}

    __host__ __device__
    void inline apply_forces(float* rho, float* u, float* force, int node) {
        float N_half = NY/2;
        force[2*node]   = 8.0f * vis * u_max / (N_half * N_half);
        force[2*node+1] = 0.0f;
    }
    
    __host__ __device__
    void operator()(float* rho, float* u, float* force, int node) {
        int x = node % NX;
        int y = node / NX;
        
        rho[node] = 1.0f;
        
        float y_normalized = abs(y - NY/2) / (float)(NY/2);
        
        float base_velocity = u_max * (1.0f - y_normalized * y_normalized);

        unsigned int seed = (x * 1103515245 + y * 12345) ^ 0xBADF00D;
        seed = (seed ^ (seed >> 16)) * 0x45d9f3b;
        seed = (seed ^ (seed >> 16)) * 0x45d9f3b;
        float random_value_x = (float)(seed & 0xFFFF) / 65536.0f - 0.5f;
        
        seed = (seed * 1103515245 + 12345) ^ 0xCAFEBABE;
        seed = (seed ^ (seed >> 16)) * 0x45d9f3b;
        seed = (seed ^ (seed >> 16)) * 0x45d9f3b;
        float random_value_y = (float)(seed & 0xFFFF) / 65536.0f - 0.5f;
        
        u[2*node]   = base_velocity + perturbation * random_value_x;
        u[2*node+1] = perturbation * random_value_y;
        
        apply_forces(rho, u, force, node);
    }
};

struct TurbulentChannelBoundary {
    __host__ __device__
    int operator()(int x, int y) {
        if (y == 0 || y == NY-1)
            return BC_flag::BOUNCE_BACK;
            
        return BC_flag::FLUID;
    }
};

struct TurbulentChannelValidation {
    // von karman constant
    static constexpr float kappa = 0.41f;

    // law of wall empirical constant
    static constexpr float B = 5.5f;
};

#endif // !TURBULENT_CHANNEL_FUNCTORS_H