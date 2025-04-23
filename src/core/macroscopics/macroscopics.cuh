#ifndef LBM_MACROSCOPICS_H
#define LBM_MACROSCOPICS_H

#include "core/lbm.cuh"
#include "core/lbm_constants.cuh"

template<int dim>
__global__ void uncorrected_macroscopics_kernel(float* f, float* rho, float* u, float* force, float* pi_mag);

template<int dim>
__global__ void correct_macroscopics_kernel(float* f, float* rho, float* u, float* force, float* pi_mag);

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
template <int dim>
__global__ void update_avg_mag(float* rho, float* u, float* pi_mag, MomentInfo* moment_info) {
    extern __shared__ float sdata[];
    
    float* s_rho    = sdata;
    float* s_j_mag  = sdata + blockDim.x * blockDim.y * blockDim.z;
    float* s_pi_mag = sdata + 2 * blockDim.x * blockDim.y * blockDim.z;
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    int tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    
    float local_rho    = 0.0f;
    float local_j_mag  = 0.0f;
    float local_pi_mag = 0.0f;
    
    if (x < NX && y < NY && z < NZ) {
        int node = z * NX * NY + y * NX + x;
        
        if constexpr (dim == 2) {
            float ux = u[get_vec_index(node, 0)];
            float uy = u[get_vec_index(node, 1)];
            
            local_rho    = rho[node];
            local_j_mag  = local_rho * sqrtf(ux*ux + uy*uy);
            local_pi_mag = pi_mag[node];
        } 
        else {
            float ux = u[get_vec_index(node, 0)];
            float uy = u[get_vec_index(node, 1)];
            float uz = u[get_vec_index(node, 2)];
            
            local_rho    = rho[node];
            local_j_mag  = local_rho * sqrtf(ux*ux + uy*uy + uz*uz);
            local_pi_mag = pi_mag[node];
        }
    }
    
    s_rho[tid]    = local_rho;
    s_j_mag[tid]  = local_j_mag;
    s_pi_mag[tid] = local_pi_mag;
    
    __syncthreads(); // all threads finish storing their magnitudes
    
    // tree-based reduction
    int block_size = blockDim.x * blockDim.y * blockDim.z;
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_rho[tid]    += s_rho[tid + s];
            s_j_mag[tid]  += s_j_mag[tid + s];
            s_pi_mag[tid] += s_pi_mag[tid + s];
        }
        __syncthreads();  // all threads finish summation
    }
    
    if (tid == 0) {
        float domain_size;
        if constexpr (dim == 2) {
            domain_size = NX * NY;
        } else {
            domain_size = NX * NY * NZ;
        }
        
        atomicAdd(&(moment_info->rho_avg_norm), (s_rho[0] / domain_size));
        atomicAdd(&(moment_info->j_avg_norm),   (s_j_mag[0] / domain_size));
        atomicAdd(&(moment_info->pi_avg_norm),  (s_pi_mag[0] / domain_size));
    }
}

template <int dim>
void LBM<dim>::uncorrected_macroscopics() {
    dim3 blocks, threads;
    if constexpr (dim == 2) {
        blocks = dim3((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (NY + BLOCK_SIZE - 1) / BLOCK_SIZE);
        threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
    }
    else {
        threads = dim3(8, 8, 4);
        blocks = dim3((NX + threads.x - 1) / threads.x,
                    (NY + threads.y - 1) / threads.y,
                    (NZ + threads.z - 1) / threads.z);
    }

    uncorrected_macroscopics_kernel<dim><<<blocks, threads>>>(d_f, d_rho, d_u, d_force, d_pi_mag);
    checkCudaErrors(cudaDeviceSynchronize());
}

template <int dim>
void LBM<dim>::correct_macroscopics() {
    dim3 blocks, threads;

    if constexpr (dim == 2) {
        blocks = dim3((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (NY + BLOCK_SIZE - 1) / BLOCK_SIZE);
        threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
    }
    else {
        threads = dim3(8, 8, 4);
        blocks = dim3((NX + threads.x - 1) / threads.x,
                    (NY + threads.y - 1) / threads.y,
                    (NZ + threads.z - 1) / threads.z);
    }

    correct_macroscopics_kernel<dim><<<blocks, threads>>>(d_f, d_rho, d_u, d_force, d_pi_mag);
    checkCudaErrors(cudaDeviceSynchronize());


    MomentInfo h_moment = {0.0f, 0.0f, 0.0f};
    checkCudaErrors(cudaMemcpyToSymbol(d_moment_avg, &h_moment, sizeof(MomentInfo)));
    
    MomentInfo* d_moment_avg_ptr;
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_moment_avg_ptr, d_moment_avg));
    
    size_t shared_size;
    if constexpr (dim == 2)
        shared_size = 3 * sizeof(float) * BLOCK_SIZE * BLOCK_SIZE;
    else
        shared_size = 3 * sizeof(float) * 8 * 8 * 4;


    update_avg_mag<dim><<<blocks, threads, shared_size>>>(d_rho, d_u, d_pi_mag, d_moment_avg_ptr);
    
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpyFromSymbol(&h_moment, d_moment_avg, sizeof(MomentInfo)));
    
    if (timestep % 100 == 0) {
        printf("[macroscopics] Moment Averages: rho_avg_norm=%.6f, momentum_avg_norm=%.6f, pi_avg_norm=%.6f\n",
            h_moment.rho_avg_norm, h_moment.j_avg_norm, h_moment.pi_avg_norm);
    }
}

#endif // ! LBM_MACROSCOPICS_H