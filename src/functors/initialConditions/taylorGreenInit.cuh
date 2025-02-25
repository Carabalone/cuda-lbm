#ifndef TAYLOR_GREEN_H
#define TAYLOR_GREEN_H

#include "lbm_constants.cuh"
#include "defines.hpp"

struct TaylorGreenInit {
    float nu;

    __host__ __device__
    TaylorGreenInit(float v) : nu(v) { }

    __device__
    inline void operator()(float* rho, float* u, float* force, int node) {
        printf("viscosity: %.4f", vis);
        const float x = (node % NX) + 0.5f; 
        const float y = (node / NX) + 0.5f;
        const float u_max = 0.04f/SCALE;
        const float rho0 = 1.0f;
        const float kx = 2.0f * M_PI / NX;
        const float ky = 2.0f * M_PI / NY;
        const float td = 1.0f / (nu * (kx*kx + ky*ky));
        const float t = 0.0f; // I could put this as arg to validate against other cases as well.
        
        float ux = -u_max * sqrt(ky/kx) * cos(kx * x) * sin(ky * y) * exp(-t / td);
        float uy = u_max * sqrt(kx/ky) * sin(kx * x) * cos(ky * y) * exp(-t / td);
        
        float P = -0.25f * rho0 * u_max * u_max * 
                  ((ky/kx)*cos(2*kx*x) + (kx/ky)*cos(2*ky*y)) * exp(-2*t/td);

        rho[node] = rho0 + 3.0f * P; // p = cs²(ρ - ρ0)
        u[2 * node] = ux;
        u[2 * node + 1] = uy;

        force[2 * node] = 0.0f;
        force[2 * node + 1] = 0.0f;
        
    }

};

#endif // ! TAYLOR_GREEN_H
