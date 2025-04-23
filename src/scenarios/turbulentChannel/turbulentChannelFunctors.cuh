#ifndef TURBULENT_CHANNEL_FUNCTORS_H
#define TURBULENT_CHANNEL_FUNCTORS_H

#include "core/lbm_constants.cuh"
#include "defines.hpp"

struct TurbulentChannelInit {
    float u_init;
    float u_tau; // wall velocity
    float perturbation;

    
    TurbulentChannelInit(float u_init, float u_tau, float perturbation) 
        : u_init(u_init), u_tau(u_tau), perturbation(perturbation) {}

    __host__ __device__
    void inline apply_forces(float* rho, float* u, float* force, int node) {
        // F_x = (u_tau)²/δ
        constexpr int half_y = NY / 2;
        float f_x = (u_tau * u_tau) / half_y;
        // float f_x = 8.0f * vis * u_tau / (NY*NY);
        
        force[get_vec_index(node, 0)] = f_x;
        force[get_vec_index(node, 1)] = 0.0f;
        force[get_vec_index(node, 2)] = 0.0f;
    }
    
    __device__
    void operator()(float* rho, float* u, float* force, int node) {
        int x, y, z;
        get_coords_from_node(node, x, y, z);

        rho[node] = 1.0f;

        float y_norm = (float)y / (NY-1);
        float parabolic = 4.0f * y_norm * (1.0f - y_norm);

        curandState state;
        curand_init(node, 0, 0, &state);

        float rand_x = gpu_rand(state, -0.5f, 0.5f);
        float rand_y = gpu_rand(state, -0.5f, 0.5f);
        float rand_z = gpu_rand(state, -0.5f, 0.5f);

        u[get_vec_index(node, 0)] = u_init * parabolic * (1.0f + perturbation * rand_x);
        u[get_vec_index(node, 1)] = u_init * perturbation * rand_y;
        u[get_vec_index(node, 2)] = u_init * perturbation * rand_z;

        // u[get_vec_index(node, 0)] = 0.0f;
        // u[get_vec_index(node, 1)] = 0.0f;
        // u[get_vec_index(node, 2)] = 0.0f;
        
        apply_forces(rho, u, force, node);
    }
};

struct TurbulentChannelBoundary {
    __host__ __device__
    int operator()(int x, int y, int z) {
        if (y == 0 || y == NY-1)
            return BC_flag::BOUNCE_BACK;
            
        return BC_flag::FLUID;
    }
};

struct TurbulentChannelValidation {
    // law of wall empirical constant
    static constexpr float B = 5.5f;

    __host__ __device__
    TurbulentChannelValidation() {}
    
    __host__ __device__
    void operator()(int x_node, int y_node, int z_node) const {
        // TODO
    }
};

#endif // !TURBULENT_CHANNEL_FUNCTORS_H