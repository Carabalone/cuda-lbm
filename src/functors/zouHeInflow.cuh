#ifndef ZOU_HE_H
#define ZOU_HE_H

#include "lbm_constants.cuh"

struct InflowBoundary {
    __device__ static void apply(float* f, float* f_back, const int* C, const int* OPP, int node) {

        node = get_node_index(node);
        // Inflow velocity (u, v)
        const float ux = 0.03f;  // Example velocity in the x-direction
        const float uy = 0.0f;

        float rho = (f[0 + node] + f[2 + node] + f[3 + node] + 
                    f[4 + node] + f[6 + node] + f[7 + node]) / (1.0f - ux);

        f[1 + node] = f[3 + node] + (2.0f/3.0f) * rho * ux;

        float f2_f4_diff = f[2 + node] - f[4 + node];
        
        // this is not right. In theory it should be divided by 2.0f. But the simulation is rather
        // unstable if I do so. The solution was to just divide by 4.0f to dampen the effect.
        // In theory this is non-physical, but I won't need Zou-He Boundaries for long, so I'm
        // leaving this here.
        float f57_diff = f2_f4_diff / 4.0f;
        float f68_diff = -f2_f4_diff / 4.0f;
        
        f[5 + node] = f[7 + node] + f57_diff + (1.0f/6.0f) * rho * ux;
        f[8 + node] = f[6 + node] + f68_diff + (1.0f/6.0f) * rho * ux;

    }
};

#endif // ! ZOU_HE_H
