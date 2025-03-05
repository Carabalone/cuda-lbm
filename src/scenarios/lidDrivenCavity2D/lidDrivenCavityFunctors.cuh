#ifndef LID_DRIVEN_CAVITY_FUNCTORS_H
#define LID_DRIVEN_CAVITY_FUNCTORS_H

#include "lbm_constants.cuh"
#include "defines.hpp"

struct LidDrivenInit {
    float u_lid;

    LidDrivenInit(float u_lid) : u_lid(u_lid) {}

    // TODO: maybe pass in the pointer to the specific location of u force and rho.
    // so we just need to do float* fx, float* *fy. fx = ..., *fy = ...
    // so that the scenario stays unaware of memory layout.
    // now its easy because we are using u1x u1y u2x u2y ... 
    // but its going to be harder with CSoA.
    __host__ __device__
    void operator()(float* rho, float* u, float* force, int node) {
        int y = node / NX;

        rho[node] = 1.0f;
        
        // u[2*node]   = (y == NY-1) ? u_lid : 0.0f;
        u[2*node]   = 0.0f;
        u[2*node+1] = 0.0f;

        force[2*node]   = 0.0f;
        force[2*node+1] = 0.0f;
    }

};

struct LidDrivenBoundary {

    __host__ __device__
    int operator()(int x, int y) {
        
        if (y == NY-1) {
            // if (x == 0)
            //     return BC_flag::ZOU_HE_TOP_LEFT_TOP_INFLOW;
            // if (x == NX)
            //     return BC_flag::ZOU_HE_TOP_RIGHT_TOP_INFLOW;
            
            return BC_flag::ZOU_HE_TOP;
        }
        else if (x == 0 || x == NX-1 || y == 0)
            return BC_flag::BOUNCE_BACK;

        return BC_flag::FLUID;
    }
};

struct LidDrivenValidation {
    
};


#endif // ! LID_DRIVEN_CAVITY_FUNCTORS_H