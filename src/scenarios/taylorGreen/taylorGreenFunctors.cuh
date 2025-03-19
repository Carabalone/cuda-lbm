
#ifndef TAYLOR_GREEN_FUNCTORS_H
#define TAYLOR_GREEN_FUNCTORS_H

#include "core/lbm_constants.cuh"
#include "defines.hpp"

struct TaylorGreenInit {
    float nu;
    float u_max;

    __host__ __device__
    TaylorGreenInit(float v, float _u_max) : nu(v) {
        u_max = _u_max / SCALE;
    }

    __device__
    inline void operator()(float* rho, float* u, float* force, int node) {
        // printf("viscosity: %.4f", vis);
        const float x = (node % NX) + 0.5f; 
        const float y = (node / NX) + 0.5f;
        const float rho0 = 1.0f;
        const float kx = 2.0f * M_PI / NX;
        const float ky = 2.0f * M_PI / NY;
        const float td = 1.0f / (nu * (kx*kx + ky*ky));
        const float t = 0.0f; // I could put this as arg to validate against other cases as well.
        
        float ux = -u_max * sqrt(ky/kx) * cos(kx * x) * sin(ky * y) * exp(-t / td);
        float uy =  u_max * sqrt(kx/ky) * sin(kx * x) * cos(ky * y) * exp(-t / td);
        
        float P = -0.25f * rho0 * u_max * u_max * 
                  ((ky/kx)*cos(2*kx*x) + (kx/ky)*cos(2*ky*y)) * exp(-2*t/td);

        rho[node] = rho0 + 3.0f * P; // p = cs²(ρ - ρ0)
        u[2 * node] = ux;
        u[2 * node + 1] = uy;

        // printf("u[%d] = (%.4f, %.4f)\n", node, ux, uy);

        force[2 * node] = 0.0f;
        force[2 * node + 1] = 0.0f;
    }
};

struct TaylorGreenBoundary {
    __host__ __device__
    int operator()(int x, int y) const {
        return BC_flag::FLUID;
    }
};

struct TaylorGreenValidation {
    float u0;
    float nu;
    float t;
    
    __host__ __device__
    TaylorGreenValidation(float u0, float nu, float t = 0.0f) 
        : u0(u0), nu(nu), t(t) {}
    
    __host__ __device__
    void operator()(int x_node, int y_node, float& ux, float& uy) const {

        if ((int)t % 100 == 0 && x_node ==0 && y_node ==0)
            printf("t: %.4f\n", t);
        const float x = x_node + 0.5f; 
        const float y = y_node + 0.5f;
        const float u_max = 0.04f/SCALE;
        const float kx = 2.0f * M_PI / NX;
        const float ky = 2.0f * M_PI / NY;
        const float td = 1.0f / (nu * (kx*kx + ky*ky));
        const float decay = exp(-t / td);

        ux = -u_max * sqrt(ky/kx) * cos(kx * x) * sin(ky * y) * decay;
        uy =  u_max * sqrt(kx/ky) * sin(kx * x) * cos(ky * y) * decay;
    }
    
    // 2D field for Taylor-Green
    std::vector<std::array<float, 2>> getFullField() const {
        std::vector<std::array<float, 2>> field(NX * NY);
        for (int y = 0; y < NY; y++) {
            for (int x = 0; x < NX; x++) {
                float ux, uy;
                operator()(x, y, ux, uy);
                field[y * NX + x] = {ux, uy};
            }
        }
        return field;
    }
    
    void setTime(float new_t) {
        t = new_t;
    }
};

#endif // ! TAYLOR_GREEN_FUNCTORS_H