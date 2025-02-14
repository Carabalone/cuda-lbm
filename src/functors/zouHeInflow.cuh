#ifndef ZOU_HE_H
#define ZOU_HE_H

#include "lbm_constants.cuh"

struct InflowBoundary {
    __device__ static void apply(float* f, float* f_back, const int* C, const int* OPP, int node) {

        node = get_node_index(node);
        // Inflow velocity (u, v)
        const float ux = 0.03f;  // Example velocity in the x-direction
        const float uy = 0.0f;

        float rho = (f[0 + node] + f[3 + node] + f[4 + node] +
                     2.0f * (f[6 + node] + f[2 + node] + f[5 + node])) / (1.0f - ux);

        // Zou/He velocity bounce-back correction
        f[1 + node] = f[3 + node] + (2.0f / 3.0f) * rho * ux;
        f[8 + node] = f[6 + node] + 0.5f * ((f[4 + node] - f[3 + node]) + rho * uy) - (1.0f / 6.0f) * rho * ux;
        f[7 + node] = f[5 + node] + 0.5f * ((f[3 + node] - f[4 + node]) - rho * uy) - (1.0f / 6.0f) * rho * ux;
    }
};

#endif // ! ZOU_HE_H
