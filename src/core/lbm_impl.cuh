#ifndef LBM_IMPL_H
#define LBM_IMPL_H

// This file is for common stuff that needs to be templated (defined in a .cuh file for deduction)
// and do not fit any of the other files. This theoretically could go into boundaries because I use it for the IBM
// but it is too general for that.

template <typename InitCond>
__global__
void reset_forces_kernel(float* d_rho, float* d_u, float* d_force, InitCond init) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int node = y * NX + x;

    init.apply_forces(d_rho, d_u, d_force, node);
}

template <typename Scenario>
void LBM::reset_forces() {
    dim3 blocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (NY+BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    auto init = Scenario::init();
    reset_forces_kernel<<<blocks, threads>>>(d_rho, d_u, d_force, init);
    checkCudaErrors(cudaDeviceSynchronize());
}

#endif // ! LBM_IMPL_H