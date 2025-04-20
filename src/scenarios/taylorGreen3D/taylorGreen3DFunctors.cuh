#ifndef TAYLOR_GREEN_FUNCTORS_3D_H
#define TAYLOR_GREEN_FUNCTORS_3D_H

#include "core/lbm_constants.cuh"
#include "defines.hpp"

struct TaylorGreen3DInit {
    float u_max;

    __host__ __device__
    TaylorGreen3DInit(float _u_max){
        u_max = _u_max;
    }

    __host__ __device__
    void inline apply_forces(float* rho, float* u, float* force, int node) {
        force[get_vec_index(node, 0)] = 0.0f;
        force[get_vec_index(node, 1)] = 0.0f;
        force[get_vec_index(node, 2)] = 0.0f;
    }

    __device__
    inline void operator()(float* rho, float* u, float* force, int node) {
        int x, y, z;
        get_coords_from_node(node, x, y, z);
        const float L = NX / (2.0f * M_PI); // domain is cubic
        
        const float kx = x / L;
        const float ky = y / L;
        const float kz = z / L;
        
        const float rho0 = 1.0f;
        const float ux =  u_max * sinf(kx) * cosf(ky) * cosf(kz);
        const float uy = -u_max * cosf(kx) * sinf(ky) * cosf(kz);
        const float uz = 0.0f;

        const float p = (rho0 * u_max * u_max / 16.0f) * 
            (cosf(2*kx) + cosf(2*ky)) * (2.0f + cosf(2*kz));
        
        rho[node] = rho0 + 3.0f * p;
        u[get_vec_index(node, 0)] = ux;
        u[get_vec_index(node, 1)] = uy;
        u[get_vec_index(node, 2)] = uz;
        
        apply_forces(rho, u, force, node);
    }
};

struct TaylorGreen3DBoundary {
    __host__ __device__
    int operator()(int x, int y) const {
        return BC_flag::FLUID;
    }
};

struct TaylorGreen3DValidation {
    float u_max;
    float nu;
    float t;
    
    __host__ __device__
    TaylorGreen3DValidation(float u_max, float nu, float t = 0.0f) 
        : u_max(u_max), nu(nu), t(t) {}
    

    __host__ __device__
    void operator()(int x_node, int y_node, int z_node) const {
        if ((int)t % 100 == 0 && x_node == 0 && y_node == 0 && z_node == 0)
            printf("t: %.4f\n", t);
    }
    
    void setTime(float new_t) {
        t = new_t;
    }
};

#endif // TAYLOR_GREEN_FUNCTORS_3D_H