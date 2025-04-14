#ifndef TAYLOR_GREEN_FUNCTORS_3D_H
#define TAYLOR_GREEN_FUNCTORS_3D_H

#include "core/lbm_constants.cuh"
#include "defines.hpp"

struct TaylorGreen3DInit {
    float nu;
    float u_max;

    __host__ __device__
    TaylorGreen3DInit(float v, float _u_max) : nu(v) {
        u_max = _u_max / SCALE;
    }

    __host__ __device__
    void inline apply_forces(float* rho, float* u, float* force, int node) {
        force[get_vec_index(node, 0)] = 0.0f;
        force[get_vec_index(node, 1)] = 0.0f;
        force[get_vec_index(node, 2)] = 0.0f;
    }

    __device__
    inline void operator()(float* rho, float* u, float* force, int node) {
        const int xy_plane = NX * NY;
        const int z = node / xy_plane;
        const int xy_part = node % xy_plane;
        const int y = xy_part / NX;
        const int x = xy_part % NX;
        
        const float x_pos = x + 0.5f;
        const float y_pos = y + 0.5f;
        const float z_pos = z + 0.5f;
        
        const float rho0 = 1.0f;
        const float kx = 2.0f * M_PI / NX;
        const float ky = 2.0f * M_PI / NY;
        const float kz = 2.0f * M_PI / NZ;
        const float td = 1.0f / (nu * (kx*kx + ky*ky + kz*kz));
        const float t = 0.0f;
        
        float ux = u_max * sin(kx * x_pos) * cos(ky * y_pos) * cos(kz * z_pos) * exp(-t / td);
        float uy = -u_max * cos(kx * x_pos) * sin(ky * y_pos) * cos(kz * z_pos) * exp(-t / td);
        float uz = 0.0f;
        
        float P = -0.0625f * rho0 * u_max * u_max * 
                 ((cos(2*kx*x_pos) + cos(2*ky*y_pos)) * (cos(2*kz*z_pos) + 2.0f)) * exp(-2*t/td);
        
        rho[node] = rho0 + 3.0f * P;
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
    float u0;
    float nu;
    float t;
    
    __host__ __device__
    TaylorGreen3DValidation(float u0, float nu, float t = 0.0f) 
        : u0(u0), nu(nu), t(t) {}
    
    __host__ __device__
    void operator()(int x_node, int y_node, int z_node, float& ux, float& uy, float& uz) const {
        if ((int)t % 100 == 0 && x_node == 0 && y_node == 0 && z_node == 0)
            printf("t: %.4f\n", t);
            
        const float x = x_node + 0.5f; 
        const float y = y_node + 0.5f;
        const float z = z_node + 0.5f;
        const float u_max = u0 / SCALE;
        const float kx = 2.0f * M_PI / NX;
        const float ky = 2.0f * M_PI / NY;
        const float kz = 2.0f * M_PI / NZ;
        const float td = 1.0f / (nu * (kx*kx + ky*ky + kz*kz));
        const float decay = exp(-t / td);
        
        ux = u_max * sin(kx * x) * cos(ky * y) * cos(kz * z) * decay;
        uy = -u_max * cos(kx * x) * sin(ky * y) * cos(kz * z) * decay;
        uz = 0.0f;
    }
    
    std::vector<std::array<float, 3>> getFullField() const {
        std::vector<std::array<float, 3>> field(NX * NY * NZ);
        for (int z = 0; z < NZ; z++) {
            for (int y = 0; y < NY; y++) {
                for (int x = 0; x < NX; x++) {
                    float ux, uy, uz;
                    operator()(x, y, z, ux, uy, uz);
                    int index = z * (NX * NY) + y * NX + x;
                    field[index] = {ux, uy, uz};
                }
            }
        }
        return field;
    }
    
    void setTime(float new_t) {
        t = new_t;
    }
};

#endif // TAYLOR_GREEN_FUNCTORS_3D_H