#ifndef BOUNDARIES_H
#define BOUNDARIES_H

// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// --------------------------------------BOUNDARIES-----------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------

template <typename Scenario>
__global__ void boundaries_kernel_2D(float* f, float* f_back, float* u, float rho, int* boundary_flags) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int idx = y * NX + x;

    // TODO solve the instantiation issue.
    BounceBack<2> domain_boundary(true, true);
    RegularizedBounceBack rbb(true, true);

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
            ZG_OutflowBoundary<2>::apply(f, idx);
            break;

        case BC_flag::PRESSURE_OUTLET:
            PressureOutlet::apply(f, f_back, idx);
            break;
        
        case BC_flag::REGULARIZED_INLET_TOP:
            RegularizedInlet::apply_top(f, f_back, Scenario::u_max, idx);
            break;

        case BC_flag::REGULARIZED_BOUNCE_BACK:
            rbb.apply(f, f_back, u[2*idx], u[2*idx+1], idx);
            break;
        case BC_flag::REGULARIZED_BOUNCE_BACK_CORNER:
            RegularizedCornerBounceBack::apply(f, f_back, u[2*idx], u[2*idx+1], idx);
            break;
        default:
            // Unknown flag; do nothing.
            printf("Unknown Flag %d\n", flag);
            break;
    }

}


template <typename Scenario>
__global__ void boundaries_kernel_3D(float* f, float* f_back, float* u, float* rho, int* boundary_flags) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= NX || y >= NY || z >= NZ) return;

    int idx = z * NX * NY + y * NX + x;

    BounceBack<3> domain_boundary(true, true, true);
    
    int flag = boundary_flags[idx];
    switch (flag) {
        case BC_flag::FLUID:
            // Interior node;
            break;
        case BC_flag::BOUNCE_BACK:
            domain_boundary.apply(f, f_back, idx);
            break;
        case BC_flag::REGULARIZED_BOUNCE_BACK:
            RegularizedBoundary::apply(f, idx, 0.0f, 0.0f, 0.0f);
            break;
        case BC_flag::REGULARIZED_INLET_LEFT:
            RegularizedBoundary::apply(f, idx, Scenario::u_max, 0.0f, 0.0f);
            break;
        case BC_flag::ZG_OUTFLOW:
            ZG_OutflowBoundary<3>::apply(f, idx);
            break;
        case BC_flag::EXTRAPOLATED_CORNER_EDGE:
            ExtrapolatedCornerEdgeBoundary<3>::apply(f, idx);
            break;
        case BC_flag::CORNER_EDGE_BOUNCE_BACK:
            EdgeCornerBounceBack::apply(f, idx);
            break;
        case BC_flag::GUO_VELOCITY_INLET:
            GuoVelocityInlet3D::apply(f, rho, u, idx, Scenario::u_max);
            break;
        case BC_flag::GUO_PRESSURE_OUTLET:
            GuoPressureOutlet3D::apply(f, rho, u, idx);
            break;
        case BC_flag::REGULARIZED_OUTLET:
            RegularizedOutlet::apply(f, rho, u, idx);
            break;
            
        default:
            // Unknown flag; do nothing.
            printf("Unknown Flag [%d]: %d\n", idx, flag);
            break;
    }
}

// there is a reason for doing two different kernels instead of templates.
// if I want to partially specialize a dim and scenario template, I need explicit instantiation
// however, for scenarios this is ridiculous.
// for CMs, I only have 3 adapters, and it's fine.
// so no tempaltes here.
template <int dim>
template <typename Scenario>
void LBM<dim>::apply_boundaries() {
    dim3 blocks, threads;
    
    if constexpr (dim == 2) {
        threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
        blocks = dim3((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                      (NY + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        boundaries_kernel_2D<Scenario><<<blocks, threads>>>(d_f, d_f_back, d_u, d_rho, d_boundary_flags);
    }
    else {
        threads = dim3(8, 8, 4);
        blocks = dim3((NX + threads.x - 1) / threads.x,
                      (NY + threads.y - 1) / threads.y,
                      (NZ + threads.z - 1) / threads.z);
        
        boundaries_kernel_3D<Scenario><<<blocks, threads>>>(d_f, d_f_back, d_u, d_rho, d_boundary_flags);
    }

    checkCudaErrors(cudaDeviceSynchronize());
}

// TODO: maybe to this in GPU ? functors can be device funcs, but I guess it is not worth it. This just runs once.
template <int dim>
template<typename BoundaryFunctor>
void LBM<dim>::setup_boundary_flags(BoundaryFunctor boundary_func) {
    int num_nodes = NX * NY * NZ; // NZ Defaults as 1 if using 2D.
    std::vector<int> h_boundary_flags(num_nodes, 0);
    
    if constexpr (dim == 2) {
        for (int y = 0; y < NY; y++) {
            for (int x = 0; x < NX; x++) {
                int node = y * NX + x;
                h_boundary_flags[node] = boundary_func(x, y);
            }
        }
    }
    else {
        for (int z = 0; z < NZ; z++) {
            for (int y = 0; y < NY; y++) {
                for (int x = 0; x < NX; x++) {
                    int node = z * NX * NY + y * NX + x;
                    h_boundary_flags[node] = boundary_func(x, y, z);
                }
            }
        }
    }
    
    checkCudaErrors(cudaMemcpy(d_boundary_flags, h_boundary_flags.data(), 
                    num_nodes * sizeof(int), cudaMemcpyHostToDevice));
}


#endif // ! BOUNDARIES_H
