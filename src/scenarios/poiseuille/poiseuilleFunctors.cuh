#ifndef POISEUILLE_FUNCTORS_H
#define POISEUILLE_FUNCTORS_H

#include "core/lbm_constants.cuh"
#include "defines.hpp"

struct PoiseuilleInit {
    float u_max;

    __host__ __device__
    PoiseuilleInit(float u_max) : u_max(u_max) { }

     __device__
    inline void operator()(float* rho, float* u, float* force, int node) {
        rho[node]       = 1.0f;
        u[2 * node]     = 0.0f;
        u[2 * node + 1] = 0.0f;

        // Guo forcing scheme.
        force[2 * node]     = 8.0f * vis * u_max / (NY*NY);
        force[2 * node + 1] = 0.0f;
    }
};

// TODO: I will use this later on.
struct PoiseuilleBoundary {
    __host__ __device__
    int operator()(int x, int y) const {
        if (y == 0 || y == NY-1) {
            return BC_flag::BOUNCE_BACK;
        }

        return BC_flag::FLUID;
    }
};

struct PoiseuilleValidation {
    float u_max;
    float viscosity;
    
    __host__ __device__
    PoiseuilleValidation(float u_max = 0.05f, float viscosity = 1.0f/6.0f) 
        : u_max(u_max), viscosity(viscosity) {}
    
    __host__ __device__
    void operator()(int x, int y, float& ux, float& uy) const {
        ux = ((8.0f * viscosity * u_max / (NY*NY)) / (2.0f * viscosity)) * y * (NY-y);
        uy = 0.0f;
    }
    
    // 1D profile -- no diference in y dir.
    std::vector<float> getProfile() const {
        std::vector<float> profile(NY);
        for (int y = 0; y < NY; y++) {
            float ux, uy;
            operator()(0, y, ux, uy);
            profile[y] = ux;
        }
        return profile;
    }
};

#endif // ! POISEUILLE_FUNCTORS_H