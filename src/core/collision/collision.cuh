#ifndef COLLISION_OPS_H
#define COLLISION_OPS_H

#include "core/lbm_constants.cuh"
#include "core/collision/BGK/BGK.cuh"
#include "core/collision/MRT/MRT.cuh"
#include "core/collision/CM/CM.cuh"
#include "core/collision/adapters.cuh"

// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// ----------------------------------------COLLISION----------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------

template <typename CollisionOp, int dim>
__global__ void collide_kernel(float* f, float* f_eq, float* u, float* force) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= NX || y >= NY || z >= NZ) return;
    
    int node = get_node_from_coords(x, y, z);

    if (node == DEBUG_NODE) {
        DPRINTF("[collide_kernel] Before collision (node %d):\n", node);
        for (int i = 0; i < quadratures; i++) {
            int idx = get_node_index(node, i);
            DPRINTF("    Dir %d: f[%d] = %f, f_eq = %f\n", i, idx, f[idx], f_eq[idx]);
        }
    }

    CollisionOp::apply(f, f_eq, u, force, node);

    if (node == DEBUG_NODE) {
        DPRINTF("[collide_kernel] After collision (node %d):\n", node);
        for (int i = 0; i < quadratures; i++) {
            int idx = get_node_index(node, i);
            DPRINTF("    Dir %d: f[%d] = %f\n", i, idx, f[idx]);
        }
    }
}

template <int dim>
template <typename CollisionOp>
void LBM<dim>::collide() {
    dim3 blocks, threads;
    
    if constexpr (dim == 2) {
        threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
        blocks = dim3((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (NY + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
    }
    else {
        threads = dim3(32, 4, 2);
        blocks = dim3((NX + threads.x - 1) / threads.x,
                     (NY + threads.y - 1) / threads.y,
                     (NZ + threads.z - 1) / threads.z);
    }
    collide_kernel<CollisionOp, dim><<<blocks, threads>>>(d_f, d_f_eq, d_u, d_force);
    
    checkCudaErrors(cudaDeviceSynchronize());
}

#endif // ! COLLISION_OPS_H