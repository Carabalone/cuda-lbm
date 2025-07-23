#ifndef PRESSURE_OUTLET_H
#define PRESSURE_OUTLET_H

#include "core/lbm_constants.cuh"

struct PressureOutlet {
    __device__ static void apply(float* f, float* f_back, int node) {
        int baseIdx = get_node_index(node, 0);
        
        int x = node % NX;
        int y = node / NX;
        int interior_node = y * NX + (x - 1);
        int baseIdx_interior = get_node_index(interior_node, 0);
        
        float rho_interior = 0.0f;
        for (int i = 0; i < quadratures; i++) {
            rho_interior += f[baseIdx_interior + i];
        }
        
        float ux_interior = 0.0f, uy_interior = 0.0f;
        for (int i = 0; i < quadratures; i++) {
            ux_interior += f[baseIdx_interior + i] * C[2*i];
            uy_interior += f[baseIdx_interior + i] * C[2*i+1];
        }
        ux_interior /= rho_interior;
        uy_interior /= rho_interior;
        
        float target_rho = 1.0f;
        
        for (int i = 0; i < quadratures; i++) {
            float ci_dot_u = C[2*i] * ux_interior + C[2*i+1] * uy_interior;
            float u_dot_u = ux_interior * ux_interior + uy_interior * uy_interior;
            float cs2 = 1.0f/3.0f;
            
            f[baseIdx + i] = WEIGHTS[i] * target_rho * (
                1.0f + 
                ci_dot_u / cs2 + 
                (ci_dot_u * ci_dot_u) / (2.0f * cs2 * cs2) - 
                u_dot_u / (2.0f * cs2)
            );
        }
    }
};

#endif // ! PRESSURE_OUTLET_H
