#ifndef LBM_CONSTANTS_H
#define LBM_CONSTANTS_H

#include "defines.hpp"

#ifdef D2Q9
    constexpr uint8_t dimensions = 2;
    constexpr uint8_t quadratures = 9;
    const float       cs = 1.0f / sqrt(3.0f);

    const float h_weights[] = {
        4.0f / 9.0f,
        1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
        1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
    };

    const int h_C[] = {
        0,  0,
        1,  0,
        0,  1,
       -1,  0,
        0, -1,
        1,  1,
       -1,  1,
       -1, -1,
        1, -1
    };

    const int h_OPP[] = {0, 3, 4, 1, 2, 7, 8, 5, 6};


    extern __constant__ float WEIGHTS[quadratures];
    extern __constant__ int C[quadratures * dimensions];
    extern __constant__ float vis;
    extern __constant__ float tau;
    extern __constant__ float omega;
    extern __constant__ int OPP[quadratures];
#endif

__device__ __host__ __forceinline__
int get_node_index(int node, int quadrature=0) {
    return node * quadratures + quadrature;
}

constexpr inline float viscosity_to_tau(float v) {
    return 3 * v + 0.5f;
}

constexpr inline float tau_to_viscosity(float t) {
    return (t - 0.5f) / 3.0f;
}

enum BC_flag {
    FLUID,
    BOUNCE_BACK,
    ZOU_HE_TOP,
    ZOU_HE_LEFT,
    ZOU_HE_TOP_LEFT_TOP_INFLOW,
    ZOU_HE_TOP_RIGHT_TOP_INFLOW,
    CYLINDER,
    ZG_OUTFLOW
};

#endif // LBM_CONSTANTS_H
