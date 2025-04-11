#ifndef LBM_MACROSCOPICS_H
#define LBM_MACROSCOPICS_H

#include "core/lbm_constants.cuh"

template <typename InitCond>
__global__ inline
void reset_forces_kernel(float* d_rho, float* d_u, float* d_force, InitCond init) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int node = y * NX + x;

    init.apply_forces(d_rho, d_u, d_force, node);
}

template <int dim>
template <typename Scenario>
void LBM<dim>::reset_forces() {
    dim3 blocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (NY+BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    auto init = Scenario::init();
    reset_forces_kernel<<<blocks, threads>>>(d_rho, d_u, d_force, init);
    checkCudaErrors(cudaDeviceSynchronize());
}

// From Optimizing Parallel Reduction in CUDA - Mark Harris
__global__ inline
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
    }
}

__global__ inline
void uncorrected_macroscopics_kernel(float* f, float* rho, float* u, float* force, float* pi_mag) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int node = y * NX + x;

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

    u[2 * node]     *= 1.0f / rho[node];
    u[2 * node + 1] *= 1.0f / rho[node];

    pi_mag[node] = sqrtf(pi[0] * pi[0] + 2.0f * pi[1] * pi[1] + pi[2] * pi[2]);

    if (node == DEBUG_NODE) {
        DPRINTF("[macroscopics_kernel] Node %d: rho=%f, u=(%f, %f)\n",
                node, rho[node], u[2*node], u[2*node+1]);
    }
}

__global__ inline
void correct_macroscopics_kernel(float* f, float* rho, float* u, float* force, float* d_pi_mag) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int node = y * NX + x;

    u[2*node]   += 0.5f * force[2*node]   / rho[node];
    u[2*node+1] += 0.5f * force[2*node+1] / rho[node];
}

template <int dim>
void LBM<dim>::uncorrected_macroscopics() {
    dim3 blocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (NY+BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    uncorrected_macroscopics_kernel<<<blocks, threads>>>(d_f, d_rho, d_u, d_force, d_pi_mag);
    checkCudaErrors(cudaDeviceSynchronize());
}

template <int dim>
void LBM<dim>::correct_macroscopics() {
    dim3 blocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (NY+BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    correct_macroscopics_kernel<<<blocks, threads>>>(d_f, d_rho, d_u, d_force, d_pi_mag);
    checkCudaErrors(cudaDeviceSynchronize());

    size_t shared_size = 3 * sizeof(float) * BLOCK_SIZE * BLOCK_SIZE;

    // reset moment averages
    MomentInfo h_moment = {0.0f, 0.0f, 0.0f};
    checkCudaErrors(cudaMemcpyToSymbol(d_moment_avg, &h_moment, sizeof(MomentInfo)));

    // update moment averages
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

#endif // ! LBM_MACROSCOPICS_H