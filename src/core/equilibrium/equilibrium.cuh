#ifndef EQUILIBRIUM_CUH
#define EQUILIBRIUM_CUH

#include "core/lbm.cuh"
#include "core/lbm_constants.cuh"
#include "functors/includes.cuh"

// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -------------------------------------EQUILIBRIUM-----------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------

// A note: this equilibrium function is truncated at SECOND ORDER.
// The higher order equilibrium function used in CMs is embedded in the formulas there used.
// we DO NOT NEED to compute f_eq if using CMs. (only for the init)


template <int dim>
__global__ void equilibrium_kernel(float* f_eq, const float* rho, const float* u) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= NX || y >= NY || z >= NZ) return;

    int node = z * NX * NY + y * NX + x;
    float node_rho = rho[node];
    float node_ux  = u[get_vec_index(node, 0)];
    float node_uy  = u[get_vec_index(node, 1)];
    float node_uz;

    if constexpr (dim == 2) {
        node_uz = 0.0f;
    }
    else {
        node_uz  = u[get_vec_index(node, 2)];
    }

    if (node == DEBUG_NODE) {
        DPRINTF("[equilibrium_kernel] Node %d (x=%d,y=%d): rho=%f, u=(%f,%f)\n",
                node, x, y, node_rho, node_ux, node_uy);
    }

    LBM<dim>::equilibrium_node(f_eq, node_ux, node_uy, node_uz, node_rho, node);

    // if (node == DEBUG_NODE) {
    //     DPRINTF("[equilibrium_kernel] f_eq[%d] = %f\n", get_node_index(node, 0), f_eq[get_node_index(node, 0)]);
    // }
}

template <int dim>
void LBM<dim>::compute_equilibrium() {
    dim3 threads, blocks;

    if constexpr (dim == 2) {
        threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
        blocks  = dim3((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (NY + BLOCK_SIZE - 1) / BLOCK_SIZE);
    }
    else {
        threads = dim3(8, 8, 4);
        blocks  = dim3((NX + threads.x - 1) / threads.x,
                    (NY + threads.y - 1) / threads.y,
                    (NZ + threads.z - 1) / threads.z);
    }

    equilibrium_kernel<dim><<<blocks, threads>>>(d_f_eq, d_rho, d_u);
    checkCudaErrors(cudaDeviceSynchronize());
}

__device__
void debug_equilibrium(float* f_eq, int node);

#endif  // ! EQUILIBRIUM_CUH