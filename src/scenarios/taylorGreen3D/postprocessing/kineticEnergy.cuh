#ifndef POSTPROCESSING_CUH
#define POSTPROCESSING_CUH

#include "core/lbm_constants.cuh"
#include "core/lbm.cuh"

__global__ void compute_kinetic_energy_kernel(float* rho, float* u, float* result) {
    extern __shared__ float sdata[];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    int tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    
    float local_energy = 0.0f;
    
    if (x < NX && y < NY && z < NZ) {
        int node = get_node_from_coords(x, y, z);
        float node_rho = rho[node];
        
        float ux = u[get_vec_index(node, 0)];
        float uy = u[get_vec_index(node, 1)];
        float uz = u[get_vec_index(node, 2)];
        
        local_energy = 0.5f * node_rho * (ux*ux + uy*uy + uz*uz);
    }
    
    sdata[tid] = local_energy;
    __syncthreads();
    
    int block_size = blockDim.x * blockDim.y * blockDim.z;
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

float calculate_kinetic_energy(float* rho, float* u) {
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));
    
    dim3 threads(8, 8, 4);
    dim3 blocks(
        (NX + threads.x - 1) / threads.x,
        (NY + threads.y - 1) / threads.y,
        (NZ + threads.z - 1) / threads.z
    );
    
    int shared_mem_size = threads.x * threads.y * threads.z * sizeof(float);
    
    compute_kinetic_energy_kernel<<<blocks, threads, shared_mem_size>>>(
        rho, u, d_result);
    cudaDeviceSynchronize();
    
    float total_energy = 0.0f;
    cudaMemcpy(&total_energy, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return total_energy / (NX * NY * NZ * 0.05f * 0.05f);
}

#endif // ! POSTPROCESSING_CUH