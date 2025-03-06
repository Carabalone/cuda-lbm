#ifndef INIT_H
#define INIT_H

// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// ---------------------------------------INIT----------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------

__device__ __forceinline__
void LBM::init_node(float* f, float* f_back, float* f_eq, float* rho, float* u, int node) {

    equilibrium_node(f_eq, u[2*node], u[2*node+1], rho[node], node);

    for (int q = 0; q < quadratures; q++) {
        int i = get_node_index(node, q);
        f[i] = f_eq[i];
        f_back[i] = f_eq[i];
    }

}

template<typename InitCond>
__global__ void init_kernel(float* f, float* f_back, float* f_eq, float* rho, float* u,
                            float* force, InitCond init, int* debug_counter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int idx = y * NX + x;

    init(rho, u, force, idx);
    atomicAdd(debug_counter, 1);
    LBM::init_node(f, f_back, f_eq, rho, u, idx);
}

template<typename Scenario>
void LBM::init() {
    dim3 blocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (NY+BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    LBM_DEVICE_ASSERT(Scenario::viscosity > 0.0f, "Negative Viscosity");
    LBM_DEVICE_ASSERT(Scenario::tau > 0.5f, "Instability warning: tau < 0.5");
    LBM_DEVICE_ASSERT(Scenario::u_max < 0.5f, "Instability warning: u_max > 0.5");

    auto init = Scenario::init();
    auto boundary_func = Scenario::boundary();

    send_consts<Scenario>();

    int* d_debug_counter;
    int h_debug_counter = 0;
    cudaMalloc(&d_debug_counter, 1 * sizeof(int));
    cudaMemcpy(d_debug_counter, &h_debug_counter, 1 * sizeof(int), cudaMemcpyHostToDevice);

    init_kernel<<<blocks, threads>>>(d_f, d_f_back, d_f_eq, d_rho, d_u, 
                                     d_force, init, d_debug_counter);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaMemcpy(&h_debug_counter, d_debug_counter, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_debug_counter);

    setup_boundary_flags(boundary_func);

    printf("[init_kernel]: Threads executed: %d\n", h_debug_counter);
}

#endif // ! INIT_H