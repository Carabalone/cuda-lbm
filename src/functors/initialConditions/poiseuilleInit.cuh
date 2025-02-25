#ifndef POISEUILLE_INIT_H
#define POISEUILLE_INIT_H

#include "lbm_constants.cuh"
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

#endif // ! POISEUILLE_INIT_H
