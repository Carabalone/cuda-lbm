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

__device__ MomentInfo d_moment_avg = {0.0f, 0.0f, 0.0f};

// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// ------------------------------------MACROSCOPICS-----------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------

__global__ void macroscopics_kernel(float* f, float* rho, float* u, float* force, float* d_pi_mag, float* d_u_uncorrected) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int node = y * NX + x;
    LBM::macroscopics_node(f, rho, u, force, d_pi_mag, d_u_uncorrected, node);
    if (node == DEBUG_NODE) {
        DPRINTF("[macroscopics_kernel] Node %d: rho=%f, u=(%f, %f)\n",
                node, rho[node], u[2*node], u[2*node+1]);
    }
}

// from Optimizing Parallel Reduction in CUDA - Mark Harris
__global__
void update_avg_mag(float* rho, float* u, float* pi_mag, MomentInfo* moment_info) {
    extern __shared__ float sdata[];
    float* s_rho    = sdata; 
    float* s_j_mag  = sdata + BLOCK_SIZE * BLOCK_SIZE;
    float* s_pi_mag = sdata + 2 * BLOCK_SIZE * BLOCK_SIZE;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // idx in block

    float local_rho    = 0.0f;
    float local_j_mag  = 0.0f; // j and not u because we keep the rho * u instead of pure u.
    float local_pi_mag = 0.0f;

    if (x < NX && y < NY) {
        int node = y * NX + x;
        float ux = u[2 * node];
        float uy = u[2 * node + 1];

        local_rho    = rho[node];
        local_j_mag  = local_rho * sqrtf(ux * ux + uy * uy);
        local_pi_mag = pi_mag[node];
    }

    s_rho[tid]    = local_rho;
    s_j_mag[tid]  = local_j_mag;
    s_pi_mag[tid] = local_pi_mag;
    // printf("accessing %d\n", tid);
    __syncthreads(); // all threads finish storing their magnitudes

    // tree-based summation
    for (int s = BLOCK_SIZE * BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_rho[tid]    += s_rho[tid + s];
            s_j_mag[tid]  += s_j_mag[tid + s];
            s_pi_mag[tid] += s_pi_mag[tid + s];
        }
        __syncthreads(); // all threads finish summation
    }

    if (tid == 0) {
        atomicAdd(&(moment_info->rho_avg_norm), (s_rho[0]    / (NX*NY)));
        atomicAdd(&(moment_info->j_avg_norm),   (s_j_mag[0]  / (NX*NY)));
        atomicAdd(&(moment_info->pi_avg_norm),  (s_pi_mag[0] / (NX*NY)));

        // printf("[update_avg_mag] Block (%d, %d): rho_avg_norm=%.6f, momentum_avg_norm=%.6f, pi_avg_norm=%.6f\n",
        //        blockIdx.x, blockIdx.y,
        //        moment_info->rho_avg_norm,
        //        moment_info->momentum_avg_norm,
        //        moment_info->pi_avg_norm);
    }

}

void LBM::macroscopics() {
    dim3 blocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (NY+BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    macroscopics_kernel<<<blocks, threads>>>(d_f, d_rho, d_u, d_force, d_pi_mag, d_u_uncorrected);
    checkCudaErrors(cudaDeviceSynchronize());

    size_t shared_size = 3 * sizeof(float) * BLOCK_SIZE * BLOCK_SIZE;

    // reset
    MomentInfo h_moment = {0.0f, 0.0f, 0.0f};
    checkCudaErrors(cudaMemcpyToSymbol(d_moment_avg, &h_moment, sizeof(MomentInfo)));

    // updating every timestep, but we could do it in intervals

    MomentInfo* d_moment_avg_ptr;
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_moment_avg_ptr, d_moment_avg));

    update_avg_mag<<<blocks, threads, shared_size>>>(d_rho, d_u, d_pi_mag, d_moment_avg_ptr);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpyFromSymbol(&h_moment, d_moment_avg, sizeof(MomentInfo)));
    checkCudaErrors(cudaDeviceSynchronize());

    if (timestep % 500 == 0 )
        printf("[macroscopics] Moment Averages: rho_avg_norm=%.6f, momentum_avg_norm=%.6f, pi_avg_norm=%.6f\n",
            h_moment.rho_avg_norm, h_moment.j_avg_norm, h_moment.pi_avg_norm);
}

__device__
void LBM::macroscopics_node(float* f, float* rho, float* u, float* force,
                            float* pi_mag, float* u_uncorrected, int node) {
    rho[node]       = 0.0f;
    u[2 * node]     = 0.0f;
    u[2 * node + 1] = 0.0f;

    // bad for 3d but whatever. we fix it later
    float pi[3] = {0.0f}; //pi_xy=pi_yx 

    for (int i=0; i < quadratures; i++) { 
        float f_i = f[get_node_index(node, i)];
         rho[node]       += f_i;
         u[2 * node]     += f_i * C[2*i];
         u[2 * node + 1] += f_i * C[2*i+1];
         pi[0]           += f_i * C[2*i] * C[2*i];
         pi[1]           += f_i * C[2*i] * C[2*i+1];
         pi[2]           += f_i * C[2*i+1] * C[2*i+1];
    }

    u_uncorrected[2*node] = u[2*node];
    u_uncorrected[2*node+1] = u[2*node+1];
    u[2 * node]     += 0.5f * force[2 * node];
    u[2 * node + 1] += 0.5f * force[2 * node + 1];

    u[2 * node]     *= 1.0f / rho[node];
    u[2 * node + 1] *= 1.0f / rho[node];

    pi_mag[node] = sqrtf(pi[0] * pi[0] + 2.0f * pi[1] * pi[1] + pi[2] * pi[2]);

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
            // printf("[WARNING][stream_node] Pushing negative/large value: "
            //     "Node (x=%3d, y=%3d) is pushing f[%d]=% .6f in Dir %d to neighbor at (x=%3d, y=%3d)\n",
            //     x, y, i, f[baseIdx + i], i, x_neigh, y_neigh);
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

// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -----------------------------------IMMERSED BOUNDARY-------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
