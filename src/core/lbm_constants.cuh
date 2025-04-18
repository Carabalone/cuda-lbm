#ifndef LBM_CONSTANTS_H
#define LBM_CONSTANTS_H

#include "defines.hpp"

#ifdef D2Q9
    constexpr uint8_t dimensions = 2;
    constexpr uint8_t quadratures = 9;
    const float       cs  = 1.0f / sqrt(3.0f);
    const float       cs2 = 1.0f / 3.0f;

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

#endif


#ifdef D3Q27
    constexpr uint8_t dimensions  = 3;
    constexpr uint8_t quadratures = 27;
    const float       cs  = 1.0f / sqrt(3.0f);
    const float       cs2 = 1.0f / 3.0f;

    const float h_weights[] = {
        8.0f/27.0f,
        2.0f/27.0f, 2.0f/27.0f, 2.0f/27.0f, 2.0f/27.0f, 2.0f/27.0f, 2.0f/27.0f,
        1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 
        1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f, 1.0f/54.0f,
        1.0f/216.0f, 1.0f/216.0f, 1.0f/216.0f, 1.0f/216.0f, 1.0f/216.0f, 1.0f/216.0f, 
        1.0f/216.0f, 1.0f/216.0f
    };

    const int h_C[] = {
        0,  0,  0,  //  - center
        1,  0,  0,  //  - face neighbors
        -1, 0,  0,  // 
        0,  1,  0,  // 
        0, -1,  0,  // 
        0,  0,  1,  // 
        0,  0, -1,  // 
        1,  1,  0,  //  - edge neighbors
        -1, 1,  0,  // 
        1, -1,  0,  // 
        -1,-1,  0,  // 
        1,  0,  1,  // 
        -1, 0,  1,  // 
        1,  0, -1,  // 
        -1, 0, -1,  // 
        0,  1,  1,  // 
        0, -1,  1,  // 
        0,  1, -1,  // 
        0, -1, -1,  // 
        1,  1,  1,  //  - corner neighbors
        -1, 1,  1,  // 
        1, -1,  1,  // 
        -1,-1,  1,  // 
        1,  1, -1,  // 
        -1, 1, -1,  // 
        1, -1, -1,  // 
        -1,-1, -1   // 
    };

    const int h_OPP[] = {
        0,                               // center
        2, 1,                            // x-axis
        4, 3,                            // y-axis
        6, 5,                            // z-axis
        10, 9, 8, 7,                     // xy plane edges
        14, 13, 12, 11,                  // xz plane edges
        18, 17, 16, 15,                  // yz plane edges
        22, 21, 20, 19, 26, 25, 24, 23   // corner neighbors
    };

    const float h_M[quadratures * quadratures] = {0.0f};

    const float h_M_inv[quadratures * quadratures] = {0.0f};

#endif

extern __constant__ float WEIGHTS[quadratures];
extern __constant__ int C[quadratures * dimensions];
extern __constant__ float vis;
extern __constant__ float tau;
extern __constant__ float omega;
extern __constant__ int OPP[quadratures];

extern __constant__ float M[quadratures*quadratures];
extern __constant__ float M_inv[quadratures*quadratures];
extern __constant__ float S[quadratures];


__device__ __host__ __forceinline__
int get_node_index(int node, int quadrature=0) {
    return node * quadratures + quadrature;
}

__device__ __host__ __forceinline__
int get_vec_index(int node, int component) {
    return node * dimensions + component;
}

__device__ __host__ __forceinline__
int get_node_from_coords(int x, int y, int z=0) {
#if defined(D2Q9)
    return y * NX + x;
#elif defined(D3Q27)
    return z * NX * NY + y * NX + x;
#endif
}

__device__ __host__ __forceinline__
void get_coords_from_node(int node, int& x, int& y, int& z) {
#if defined(D2Q9)
    x = node % NX;
    y = node / NX;
    z = 0;
#elif defined(D3Q27)
    x = node % NX;
    y = (node / NX) % NY;
    z = node / (NX * NY);
#endif
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
