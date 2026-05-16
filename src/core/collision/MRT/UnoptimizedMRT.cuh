#ifndef LBM_COLLISION_UNOPTIMIZED_MRT_H
#define LBM_COLLISION_UNOPTIMIZED_MRT_H

#include "core/lbm.cuh"
#include "core/lbm_constants.cuh"
#include "core/collision/MRT/MRT.cuh"

// Slow reference MRT: full M / M_inv matrix multiplies and equilibrium from f_eq.
// Intended as a baseline (pair with AoS via defines.hpp).
template <int dim>
struct UnoptimizedMRT {
    __device__ __forceinline__ static
    void apply(float* f, float* f_eq, float* u, float* force, int node);
};

#endif // ! LBM_COLLISION_UNOPTIMIZED_MRT_H
