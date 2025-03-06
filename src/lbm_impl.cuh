#ifndef LBM_IMPL_H
#define LBM_IMPL_H

// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// ---------------------------------------INIT----------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------

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

    LBM_ASSERT(Scenario::viscosity > 0.0f, "Negative Viscosity");
    LBM_ASSERT(Scenario::tau > 0.5f, "Instability warning: tau < 0.5");
    LBM_ASSERT(Scenario::u_max < 0.5f, "Instability warning: u_max > 0.5");

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

// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// --------------------------------------BOUNDARIES-----------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------

template <typename Scenario>
__global__ void boundaries_kernel(float* f, float* f_back, int* boundary_flags) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int idx = y * NX + x;

    BBDomainBoundary domain_boundary(true, true);

    int flag = boundary_flags[idx];
    switch (flag) {
        case BC_flag::FLUID:
            // Interior node;
            break;
        case BC_flag::BOUNCE_BACK:
            // Top/bottom bounce-back
            domain_boundary.apply(f, f_back, idx);
            break;

        case BC_flag::ZOU_HE_TOP:
            ZouHe::apply_top(f, f_back, Scenario::u_max, idx);
            break;

        case BC_flag::ZOU_HE_LEFT:
            ZouHe::apply_left(f, f_back, Scenario::u_max, idx);
            break;

        case BC_flag::ZOU_HE_TOP_LEFT_TOP_INFLOW:
            // ZouHe::apply_top_left_corner(f, f_back, idx);
            printf("[NOT_IMPLEMENTED] Zou/He Top-left (top inflow)\n");
            break;

        case BC_flag::ZOU_HE_TOP_RIGHT_TOP_INFLOW:
            // ZouHe::apply_top_right_corner(f, f_back, idx);
            printf("[NOT_IMPLEMENTED] Zou/He Top-right (top inflow)\n");
            break;

        case BC_flag::CYLINDER:
            CylinderBoundary::apply(f, f_back, idx);
            break;

        case BC_flag::ZG_OUTFLOW:
            OutflowBoundary::apply(f, f_back, idx);
            break;

        default:
            // Unknown flag; do nothing.
            printf("Unknown Flag\n");
            break;
    }

}

template <typename Scenario>
void LBM::apply_boundaries() {
    dim3 blocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (NY+BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    boundaries_kernel<Scenario><<<blocks, threads>>>(d_f, d_f_back, d_boundary_flags);
    checkCudaErrors(cudaDeviceSynchronize());
}

template<typename BoundaryFunctor>
void LBM::setup_boundary_flags(BoundaryFunctor boundary_func) {
    int num_nodes = NX * NY;
    std::vector<int> h_boundary_flags(num_nodes, 0);
    
    for (int y = 0; y < NY; y++) {
        for (int x = 0; x < NX; x++) {
            int node = y * NX + x;
            h_boundary_flags[node] = boundary_func(x, y);
        }
    }
    
    checkCudaErrors(cudaMemcpy(d_boundary_flags, h_boundary_flags.data(), 
                    num_nodes * sizeof(int), cudaMemcpyHostToDevice));
}


#endif // ! LBM_IMPL_H