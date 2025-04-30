#ifndef LBM_COLLISION_MRT_H
#define LBM_COLLISION_MRT_H

#include "core/lbm.cuh"
#include "core/lbm_constants.cuh"

template <int dim>
struct MRT {
    __device__ __forceinline__ static
    void apply(float* f, float* f_eq, float* u, float* force, int node);

    __device__ static
    void compute_forcing_term(float* F, float* u, float* force, int node);
};

// template struct MRT<2>;
// template struct MRT<3>;

#endif // ! LBM_COLLISION_MRT_H