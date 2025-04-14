#ifndef LBM_STREAMING_H
#define LBM_STREAMING_H

#define PERIODIC
#define PERIODIC_X
#define PERIODIC_Y
// #define PERIODIC_Z

template <int dim>
__global__ inline void stream_kernel(float* f, float* f_back) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int node = y * NX + x;

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
__device__ inline
void LBM<dim>::stream_node(float* f, float* f_back, int node) {
    const int x = node % NX;
    const int y = node / NX;
    const int baseIdx = get_node_index(node, 0);

    for (int i=1; i < quadratures; i++) {
        int x_neigh = x + C[2*i];
        int y_neigh = y + C[2*i+1];

#ifndef PERIODIC
        if (x_neigh < 0 || x_neigh >= NX || y_neigh < 0 || y_neigh >= NY)
            continue;
#else
    #ifdef PERIODIC_X
        #ifndef PERIODIC_Y
        if (y_neigh < 0 || y_neigh >= NY)
            continue;
        #endif
        x_neigh = (x_neigh + NX) % NX;
    #endif
    #ifdef PERIODIC_Y
        #ifndef PERIODIC_X
        if (x_neigh < 0 || x_neigh >= NX)
            continue;
        #endif
        y_neigh = (y_neigh + NY) % NY;
    #endif
#endif

        const int idx_neigh = get_node_index(NX * y_neigh + x_neigh, i);
        float source_val = f[baseIdx + i];

        f_back[idx_neigh] = f[baseIdx + i];

        if (fabsf(f_back[idx_neigh]) > VALUE_THRESHOLD || f_back[idx_neigh] < -0.01f) {
            // printf("[WARNING][stream_node] Pushing negative/large value: "
            //     "Node (x=%3d, y=%3d) is pushing f[%d]=% .6f in Dir %d to neighbor at (x=%3d, y=%3d)\n",
            //     x, y, i, f[baseIdx + i], i, x_neigh, y_neigh);
        }
    }
}

template <int dim>
void LBM<dim>::stream() {
    dim3 blocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (NY+BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    stream_kernel<dim><<<blocks, threads>>>(d_f, d_f_back);
    checkCudaErrors(cudaDeviceSynchronize());
}

#endif // ! LBM_STREAMING_H