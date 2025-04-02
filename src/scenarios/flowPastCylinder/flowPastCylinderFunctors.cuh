#ifndef FLOW_PAST_CYLINDER_FUNCTORS_H
#define FLOW_PAST_CYLINDER_FUNCTORS_H

#include "core/lbm_constants.cuh"
#include "defines.hpp"
#include "functors/includes.cuh"

struct FlowPastCylinderInit {
    float u_max;
    float cx, cy, r;
    
    FlowPastCylinderInit(float u_max, float cx, float cy, float r) 
        : u_max(u_max), cx(cx), cy(cy), r(r) {}
    
    // This seems arbitrary, but it is EXTREMELY important to add this apply forces function if you plan to use the IBM
    // This function will be called when we need to reset the forces after an IBM step.
    // Even if you do not plan to use it, it is still good practice.
    // Without it you will just get an empty array instead of an array with your forces.
    __host__ __device__
    void inline apply_forces(float* rho, float* u, float* force, int node) {
        force[2*node]   = 0.0f;
        force[2*node+1] = 0.0f;
    }
    
    __host__ __device__
    void operator()(float* rho, float* u, float* force, int node) {
        int x = node % NX;
        int y = node / NX;
        
        rho[node] = 1.0f;

        float dx = x - cx;
        float dy = y - cy;
        bool is_inside = (dx * dx + dy * dy) <= (r * r);
        
        if (is_inside) {
            u[2*node] = 0.0f;
            u[2*node+1] = 0.0f;
        } else {
            // u[2*node] = 0.5 * u_max * (1.0f - exp(-static_cast<float>(x) / 20.0f));
            // u[2*node+1] = 0.0f;
        }

        apply_forces(rho, u, force, node);
    }
};

struct FlowPastCylinderBoundary {
    float cx, cy, r;
    
    FlowPastCylinderBoundary(float cx, float cy, float r) 
        : cx(cx), cy(cy), r(r) {}
    
    __host__ __device__
    int operator()(int x, int y) {
        // Top and bottom walls: bounce back
        if (y == 0 || y == NY-1)
            return BC_flag::BOUNCE_BACK;
            
        // Left boundary: Zou/He velocity inlet
        if (x == 0)
            return BC_flag::ZOU_HE_LEFT;
            
        // Right boundary: outflow
        if (x == NX-1)
            return BC_flag::ZG_OUTFLOW;
        
        // Check if node is inside cylinder
        float dx = x - cx;
        float dy = y - cy;
        // use ibm now
        // if ((dx * dx + dy * dy) <= (r * r))
        //     return BC_flag::CYLINDER;
            
        // Default: fluid node
        return BC_flag::FLUID;
    }
};

struct FlowPastCylinderValidation {
    // No analytical solution for comparison
};

#endif // !FLOW_PAST_CYLINDER_FUNCTORS_H