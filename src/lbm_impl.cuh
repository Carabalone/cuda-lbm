#ifndef LBM_IMPL_H
#define LBM_IMPL_H

template<typename InitCond>
__global__ void init_kernel(float* f, float* f_back, float* f_eq, float* rho, float* u,
                            InitCond init, int* debug_counter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int idx = y * NX + x;

    init(rho, u, idx);
    atomicAdd(debug_counter, 1);
    LBM::init_node(f, f_back, f_eq, rho, u, idx);
}

template<typename InitCond>
void LBM::init(const InitCond& init) {
    dim3 blocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (NY+BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    send_consts();

    int* d_debug_counter;
    int h_debug_counter = 0;
    cudaMalloc(&d_debug_counter, 1 * sizeof(int));
    cudaMemcpy(d_debug_counter, &h_debug_counter, 1 * sizeof(int), cudaMemcpyHostToDevice);

    init_kernel<<<blocks, threads>>>(d_f, d_f_back, d_f_eq, d_rho, d_u, 
                                     init, d_debug_counter);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaMemcpy(&h_debug_counter, d_debug_counter, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_debug_counter);

    setup_boundary_flags();

    printf("[init_kernel]: Threads executed: %d\n", h_debug_counter);
}


#endif // ! LBM_IMPL_H