#ifndef LBM_STREAMING_H
#define LBM_STREAMING_H

#include "core/lbm.cuh"
#include "core/lbm_constants.cuh"
#include "functors/includes.cuh"

#define PERIODIC
#define PERIODIC_X
#define PERIODIC_Y
// #define PERIODIC_Z

template <int dim>
__global__ inline void stream_kernel(float* f, float* f_back) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // NZ defaults as 1, z as 0
    if (x >= NX || y >= NY || z >= NZ) return;

    int node;

    if constexpr (dim == 2)
        node = y * NX + x;
    else
        node = (z * NX * NY + y * NX + x);

    if (node == DEBUG_NODE) {
        DPRINTF("[stream_kernel] Before streaming (node %d):\n", node);
        for (int i = 0; i < quadratures; i++) {
            DPRINTF("    f[%d] = %f\n", get_node_index(node, i), f[get_node_index(node, i)]);
        }
    }

    LBM<dim>::stream_node(f, f_back, node);

    if (node == DEBUG_NODE) {
        DPRINTF("[stream_kernel] After streaming (node %d):\n", node);
        for (int i = 0; i < quadratures; i++) {
            DPRINTF("    f_back[%d] = %f\n", get_node_index(node, i), f_back[get_node_index(node, i)]);
        }
    }
}

template <int dim>
void LBM<dim>::stream() {
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

    stream_kernel<dim><<<blocks, threads>>>(d_f, d_f_back);
    checkCudaErrors(cudaDeviceSynchronize());
}

#endif // ! LBM_STREAMING_H