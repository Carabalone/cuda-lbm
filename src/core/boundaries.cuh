#ifndef BOUNDARIES_H
#define BOUNDARIES_H

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

    // TODO solve the instantiation issue.
    BBDomainBoundary domain_boundary(true, true);

    static constexpr float r = NY / 12.0f;
    static constexpr float cx = NX / 4.0f;
    static constexpr float cy = NY / 2.0f;

    CylinderBoundary cb(cx, cy, r);

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
            LBM_ASSERT(false, "[NOT_IMPLEMENTED] Zou/He Top-left (top inflow)\n");
            break;

        case BC_flag::ZOU_HE_TOP_RIGHT_TOP_INFLOW:
            // ZouHe::apply_top_right_corner(f, f_back, idx);
            LBM_ASSERT(false, "[NOT_IMPLEMENTED] Zou/He Top-right (top inflow)\n");
            break;

        case BC_flag::CYLINDER:
            cb.apply(f, f_back, idx);
            break;

        case BC_flag::ZG_OUTFLOW:
            OutflowBoundary::apply(f, f_back, idx);
            break;

        case BC_flag::PRESSURE_OUTLET:
            PressureOutlet::apply(f, f_back, idx);
            break;

        default:
            // Unknown flag; do nothing.
            printf("Unknown Flag %d\n", flag);
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


#endif // ! BOUNDARIES_H