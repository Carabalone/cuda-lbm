#ifndef IBM_IMPL_CUH
#define IBM_IMPL_CUH

#include "IBM/IBMBody.cuh"
#include "IBM/IBMUtils.cuh"

template <int dim>
__global__
void compute_lagrangian_kernel(float* points, float* u_ibm, float* forces_lagrangian, int num_pts, float* rho_ibm) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= num_pts) return;

    //TODO: make this customizable if needed
    constexpr float u_target = 0.0f; // noslip boundary

    #pragma unroll
    for (int d = 0; d < dim; d++) {
        float u = u_ibm[get_lag_vec_index(idx, d, num_pts)];
        forces_lagrangian[get_lag_vec_index(idx, d, num_pts)] = 2.0f * rho_ibm[idx] * (u_target - u);
    }
}

template <int dim>
__global__
void correct_velocities_kernel(float* u, float* f_iter, float* rho) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= NX || y >= NY || z >= NZ) return;
    
    int node = z * NX * NY + y * NX + x;
    
    #pragma unroll
    for (int d = 0; d < dim; d++) {
        u[get_vec_index(node, d)] = u[get_vec_index(node, d)] + f_iter[get_vec_index(node, d)] / (2.0f * rho[node]);
    }
}

template <int dim>
__global__
void accumulate_forces_kernel(float* forces_total, float* iter_force) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= NX || y >= NY || z >= NZ) return;

    int node = z * NX * NY + y * NX + x;

    #pragma unroll
    for (int d = 0; d < dim; d++) {
        forces_total[get_vec_index(node, d)] += iter_force[get_vec_index(node, d)];
    }

    if (fabsf(forces_total[2 * node]) > 0.001f) {
            // printf("Node (%d, %d) | Force X: %.6f | Force Y: %.6f\n", 
            //     x, y, forces_total[2 * node], forces_total[2 * node + 1]);
    }
}

#endif // ! IBM_IMPL_CUH