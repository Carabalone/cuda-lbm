#ifndef LID_DRIVEN_CAVITY_FUNCTORS_H
#define LID_DRIVEN_CAVITY_FUNCTORS_H

#include "core/lbm_constants.cuh"
#include "defines.hpp"

struct LidDrivenInit {
    float u_lid;

    LidDrivenInit(float u_lid) : u_lid(u_lid) {}

    __host__ __device__
    void inline apply_forces(float* rho, float* u, float* force, int node) {
        force[2*node] = 0.0f;
        force[2*node+1] = 0.0f;
    }

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
        
        apply_forces(rho, u, force, node);
    }

};

struct LidDrivenBoundary {

    __host__ __device__
    int operator()(int x, int y) {
        
        if ((x == 0 && y == 0)    || (x == 0 && y == NY-1) ||
            (x == NX-1 && y == 0) || (x == NX-1 && y == NY-1)){
            return BC_flag::REGULARIZED_BOUNCE_BACK_CORNER;
            // return BC_flag::BOUNCE_BACK;
        }
        else if (y == NY-1) {
            return BC_flag::REGULARIZED_INLET_TOP;
        }
        else if (x == 0 || x == NX-1 || y == 0)
            // return BC_flag::BOUNCE_BACK;
            return BC_flag::REGULARIZED_BOUNCE_BACK;

        return BC_flag::FLUID;
    }
};

struct LidDrivenValidation {
    // Ghia et. al (High-Re Solutions for Incompressible Flow Using the Navier-Stokes Equations and a Multigrid Method*)
    static constexpr int ghia_u_count = 17;
    
    // --------------------------------------------------------------------------------------------
    // ----------------------------------- Re=100 ------------------------------------------------
    // --------------------------------------------------------------------------------------------

    // data for u-velocity along vertical centerline (x=0.5)
    const float ghia_y[ghia_u_count] = {
        1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 
        0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000
    };
    const float ghia_ux_100[ghia_u_count] = { // at Re=100
        1.0000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641, 
        -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.00000
    };

    // data for v-velocity along horizontal centerline (y=0.5)
    const float ghia_x[ghia_u_count] = {
        1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 
        0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000
    };
    const float ghia_uy_100[ghia_u_count] = {
        0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533, 
        0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091, 0.09233, 0.00000
    };

    // --------------------------------------------------------------------------------------------
    // ----------------------------------- Re=400 ------------------------------------------------
    // --------------------------------------------------------------------------------------------

    // data for u-velocity along vertical centerline (x=0.5)
    const float ghia_ux_400[ghia_u_count] = {
        1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135, 
        -0.11477, -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266, -0.08186, 0.00000
    };
    
    // data for v-velocity along horizontal centerline (y=0.5)
    const float ghia_uy_400[ghia_u_count] = {
        0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993, -0.38598, 
        0.05188, 0.30174, 0.30203, 0.28124, 0.22965, 0.20920, 0.19713, 0.18360, 0.00000
    };

    // --------------------------------------------------------------------------------------------
    // ----------------------------------- Re=1000 ------------------------------------------------
    // --------------------------------------------------------------------------------------------
    // data for u-velocity along vertical centerline (x=0.5)
    const float ghia_ux_1000[ghia_u_count] = {
        1.0000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702, 
        -0.06080, -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109, 0.00000
    };
    
    // data for v-velocity along horizontal centerline (y=0.5)
    const float ghia_uy_1000[ghia_u_count] = {
        0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665, -0.31966, 
        0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 0.29012, 0.27485, 0.00000
    };

    // --------------------------------------------------------------------------------------------
    // ----------------------------------- Re=3200 ------------------------------------------------
    // --------------------------------------------------------------------------------------------
    const float ghia_ux_3200[ghia_u_count] = {
        1.0000, 0.53236, 0.48296, 0.46547, 0.46101, 0.34682, 0.19791, 0.07156, 
        -0.04272, -0.86636, -0.24427, -0.34323, -0.41933, -0.37827, -0.35344, -0.32407, 0.00000
    };
    
    const float ghia_uy_3200[ghia_u_count] = {
        0.00000, -0.39017, -0.47425, -0.52357, -0.54053, -0.44307, -0.37401, -0.31184, 
        0.00999, 0.28188, 0.29030, 0.37119, 0.42768, 0.41906, 0.40917, 0.39560, 0.00000
    };

    // --------------------------------------------------------------------------------------------
    // ----------------------------------- Re=5000 ------------------------------------------------
    // --------------------------------------------------------------------------------------------
    const float ghia_ux_5000[ghia_u_count] = {
        1.0000, 0.48223, 0.46120, 0.45992, 0.46036, 0.33556, 0.20087, 0.08183, 
        -0.03039, -0.07404, -0.22855, -0.33050, -0.40435, -0.43643, -0.42901, -0.41165, 0.00000
    };
    
    const float ghia_uy_5000[ghia_u_count] = {
        0.00000, -0.49774, -0.55069, -0.55408, -0.52876, -0.41442, -0.36214, -0.30018, 
        0.00945, 0.27280, 0.28066, 0.35368, 0.42951, 0.43648, 0.43329, 0.42447, 0.00000
    };

    // --------------------------------------------------------------------------------------------
    // ----------------------------------- Re=7500 ------------------------------------------------
    // --------------------------------------------------------------------------------------------
    const float ghia_ux_7500[ghia_u_count] = {
        1.0000, 0.47244, 0.47048, 0.47323, 0.47167, 0.34228, 0.20591, 0.08342, 
        -0.03800, -0.07503, -0.23176, -0.32393, -0.38324, -0.43025, -0.43590, -0.43154, 0.00000
    };
    
    const float ghia_uy_7500[ghia_u_count] = {
        0.00000, -0.53858, -0.55216, -0.52347, -0.48590, -0.41050, -0.36213, -0.30448, 
        0.00824, 0.27348, 0.28117, 0.35060, 0.41824, 0.43564, 0.44030, 0.43979, 0.00000
    };

    // --------------------------------------------------------------------------------------------
    // ----------------------------------- Re=10000 -----------------------------------------------
    // --------------------------------------------------------------------------------------------
    const float ghia_ux_10000[ghia_u_count] = {
        1.0000, 0.47221, 0.47783, 0.48070, 0.47804, 0.34635, 0.20673, 0.08344, 
        0.03111, -0.07540, -0.23186, -0.32709, -0.38000, -0.41657, -0.42537, -0.42735, 0.00000
    };
    
    const float ghia_uy_10000[ghia_u_count] = {
        0.00000, -0.54302, -0.52987, -0.49099, -0.45863, -0.41496, -0.36737, -0.30719, 
        0.00831, 0.27224, 0.28003, 0.35070, 0.41487, 0.43124, 0.43733, 0.43983, 0.00000
    };


    const float* ux_ref(int re) const {
        switch (re) {
            case 100: return ghia_ux_100;
            case 400: return ghia_ux_400;
            case 1000: return ghia_ux_1000;
            case 3200: return ghia_ux_3200;
            case 5000: return ghia_ux_5000;
            case 7500: return ghia_ux_7500;
            case 10000: return ghia_ux_10000;
            default: 
                printf("Unsupported Reynolds number: %d. Using Re=100 data.\n", re);
                return ghia_ux_100;
        }
    }
    
    const float* uy_ref(int re) const {
        switch (re) {
            case 100: return ghia_uy_100;
            case 400: return ghia_uy_400;
            case 1000: return ghia_uy_1000;
            case 3200: return ghia_uy_3200;
            case 5000: return ghia_uy_5000;
            case 7500: return ghia_uy_7500;
            case 10000: return ghia_uy_10000;
            default: 
                printf("Unsupported Reynolds number: %d. Using Re=100 data.\n", re);
                return ghia_uy_100;
        }
    }

    const float* get_closest_ref_data(int re, bool is_ux) const {
        const int available[] = {100, 400, 1000, 3200, 5000, 7500, 10000};
        
        int closest = available[0];
        int min_diff = abs(re - available[0]);
        
        for (int i = 1; i < 7; i++) {
            int diff = abs(re - available[i]);
            if (diff < min_diff) {
                min_diff = diff;
                closest = available[i];
            }
        }
        
        // Return the appropriate data array
        if (is_ux) {
            return ux_ref(closest);
        } else {
            return uy_ref(closest);
        }
    }
};


#endif // ! LID_DRIVEN_CAVITY_FUNCTORS_H