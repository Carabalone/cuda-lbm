#include "core/lbm.cuh"
#include "core/lbm_constants.cuh"
#include "functors/includes.cuh"


// #define DEBUG_KERNEL
#define PERIODIC
#define PERIODIC_X
#define PERIODIC_Y
// #define PERIODIC_Z


__constant__ float WEIGHTS[quadratures];
__constant__ int C[quadratures * dimensions];
__constant__ float vis;
__constant__ float tau;
__constant__ float omega;
__constant__ int OPP[quadratures];

__constant__ float M[quadratures * quadratures];
__constant__ float M_inv[quadratures * quadratures];
__constant__ float S[quadratures];

__device__ MomentInfo d_moment_avg = {0.0f, 0.0f, 0.0f};

template class LBM<2>;
template class LBM<3>;

// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -----------------------------------IMMERSED BOUNDARY-------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------

