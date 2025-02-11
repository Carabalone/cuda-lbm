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
#endif

// TODO: change later, ik this is bad
constexpr float h_tau   = 0.55f;
constexpr float h_omega = 1 / h_tau;

__device__ __host__ __forceinline__
int get_node_index(int node, int quadrature=0) {
    return node * quadratures + quadrature;
}

#endif // LBM_CONSTANTS_H
