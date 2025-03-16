#include "core/lbm.cuh"
#include "core/lbm_constants.cuh"
#include "functors/includes.cuh"


// #define DEBUG_KERNEL
#define PERIODIC
#define PERIODIC_X
#define PERIODIC_Y


#ifdef D2Q9
    __constant__ float WEIGHTS[quadratures];
    __constant__ int C[quadratures * dimensions];
    __constant__ float vis;
    __constant__ float tau;
    __constant__ float omega;
    __constant__ int OPP[quadratures];

    __constant__ float M[quadratures * quadratures];
    __constant__ float M_inv[quadratures * quadratures];
    __constant__ float S[quadratures];
#endif

// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// ------------------------------------MACROSCOPICS-----------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------

__global__ void macroscopics_kernel(float* f, float* rho, float* u, float* force) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int node = y * NX + x;
    LBM::macroscopics_node(f, rho, u, force, node);
    if (node == DEBUG_NODE) {
        DPRINTF("[macroscopics_kernel] Node %d: rho=%f, u=(%f, %f)\n",
                node, rho[node], u[2*node], u[2*node+1]);
    }
}

void LBM::macroscopics() {
    dim3 blocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (NY+BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    macroscopics_kernel<<<blocks, threads>>>(d_f, d_rho, d_u, d_force);
    checkCudaErrors(cudaDeviceSynchronize());
}

__device__
void LBM::macroscopics_node(float* f, float* rho, float* u, float* force, int node) {
    rho[node]       = 0.0f;
    u[2 * node]     = 0.0f;
    u[2 * node + 1] = 0.0f;

    for (int i=0; i < quadratures; i++) { 
         // f[get_node_index(node, i)] = 
         rho[node]       += f[get_node_index(node, i)];
         u[2 * node]     += f[get_node_index(node, i)] * C[2 * i];
         u[2 * node + 1] += f[get_node_index(node, i)] * C[2 * i + 1];
    }

    u[2 * node]     += 0.5f * force[2 * node];
    u[2 * node + 1] += 0.5f * force[2 * node + 1];

    // printf("adding %f\n", (force[2 * node]));

    u[2 * node]     *= 1.0f / rho[node];
    u[2 * node + 1] *= 1.0f / rho[node];

    // if (node == 0) {
    //     printf("Node 0: rho=%.4f, ux=%.4f, uy=%.4f\n", 
    //           rho[node], u[2*node], u[2*node+1]);
    // }
}

// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// --------------------------------------STREAMING------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------

__device__
void LBM::stream_node(float* f, float* f_back, int node) {
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
            printf("[WARNING][stream_node] Pushing negative/large value: "
                "Node (x=%3d, y=%3d) is pushing f[%d]=% .6f in Dir %d to neighbor at (x=%3d, y=%3d)\n",
                x, y, i, f[baseIdx + i], i, x_neigh, y_neigh);
        }
    }
}

__global__ void stream_kernel(float* f, float* f_back) {
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

    LBM::stream_node(f, f_back, node);

    if (node == DEBUG_NODE) {
        DPRINTF("[stream_kernel] After streaming (node %d):\n", node);
        for (int i = 0; i < quadratures; i++) {
            DPRINTF("    f_back[%d] = %f\n", get_node_index(node, i), f_back[get_node_index(node, i)]);
        }
    }
}

void LBM::stream() {
    dim3 blocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (NY+BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    stream_kernel<<<blocks, threads>>>(d_f, d_f_back);
    checkCudaErrors(cudaDeviceSynchronize());

    // float* temp;
    // temp = d_f;
    // d_f = d_f_back;
    // d_f_back = temp;
}
