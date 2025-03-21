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

    const float h_M[] = {
        1,  1,  1,  1,  1,  1,  1,  1,  1,
       -4, -1, -1, -1, -1,  2,  2,  2,  2,
        4, -2, -2, -2, -2,  1,  1,  1,  1,
        0,  1,  0, -1,  0,  1, -1, -1,  1,
        0, -2,  0,  2,  0,  1, -1, -1,  1,
        0,  0,  1,  0, -1,  1,  1, -1, -1,
        0,  0, -2,  0,  2,  1,  1, -1, -1,
        0,  1, -1,  1, -1,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  1, -1,  1, -1
    };

    const float h_M_inv[] = {
        1.0f/9.0f,  -1.0f/9.0f,  1.0f/9.0f,    0.0f,       0.0f,        0.0f,        0.0f,        0.0f,       0.0f,
        1.0f/9.0f,  -1.0f/36.0f, -1.0f/18.0f,  1.0f/6.0f, -1.0f/6.0f,   0.0f,        0.0f,        1.0f/4.0f,  0.0f,
        1.0f/9.0f,  -1.0f/36.0f, -1.0f/18.0f,  0.0f,       0.0f,        1.0f/6.0f,  -1.0f/6.0f,  -1.0f/4.0f,  0.0f,
        1.0f/9.0f,  -1.0f/36.0f, -1.0f/18.0f, -1.0f/6.0f,  1.0f/6.0f,   0.0f,        0.0f,        1.0f/4.0f,  0.0f,
        1.0f/9.0f,  -1.0f/36.0f, -1.0f/18.0f,  0.0f,       0.0f,       -1.0f/6.0f,   1.0f/6.0f,  -1.0f/4.0f,  0.0f,
        1.0f/9.0f,   1.0f/18.0f,  1.0f/36.0f,  1.0f/6.0f,  1.0f/12.0f,  1.0f/6.0f,   1.0f/12.0f,  0.0f,       1.0f/4.0f,
        1.0f/9.0f,   1.0f/18.0f,  1.0f/36.0f, -1.0f/6.0f, -1.0f/12.0f,  1.0f/6.0f,   1.0f/12.0f,  0.0f,      -1.0f/4.0f,
        1.0f/9.0f,   1.0f/18.0f,  1.0f/36.0f, -1.0f/6.0f, -1.0f/12.0f, -1.0f/6.0f,  -1.0f/12.0f,  0.0f,       1.0f/4.0f,
        1.0f/9.0f,   1.0f/18.0f,  1.0f/36.0f,  1.0f/6.0f,  1.0f/12.0f, -1.0f/6.0f,  -1.0f/12.0f,  0.0f,      -1.0f/4.0f
    };

    extern __constant__ float WEIGHTS[quadratures];
    extern __constant__ int C[quadratures * dimensions];
    extern __constant__ float vis;
    extern __constant__ float tau;
    extern __constant__ float omega;
    extern __constant__ int OPP[quadratures];

    extern __constant__ float M[quadratures*quadratures];
    extern __constant__ float M_inv[quadratures*quadratures];
    extern __constant__ float S[quadratures];


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

constexpr float compute_reynolds(float u_max, float domain_size, float viscosity) {
    return (u_max * domain_size) / viscosity;
}

enum BC_flag {
    FLUID,
    BOUNCE_BACK,
    ZOU_HE_TOP,
    ZOU_HE_LEFT,
    ZOU_HE_TOP_LEFT_TOP_INFLOW,
    ZOU_HE_TOP_RIGHT_TOP_INFLOW,
    CYLINDER,
    ZG_OUTFLOW,
    PRESSURE_OUTLET,
    REGULARIZED_INLET_TOP,
    REGULARIZED_BOUNCE_BACK,
    REGULARIZED_BOUNCE_BACK_CORNER
};

#endif // LBM_CONSTANTS_H
