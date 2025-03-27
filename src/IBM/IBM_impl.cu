#include "IBM/IBMBody.cuh"
#include "IBM/IBMUtils.cuh"

IBMBody create_cylinder(float cx, float cy, float r, int num_pts) {

    IBMBody body = {num_pts, nullptr, nullptr};
    body.points = new float[2*num_pts];
    body.velocities = new float[2*num_pts];

    float angle = 2*M_PI / num_pts; float coord[2];
    for (int i=0; i < num_pts; i++) {
        coord[0] = cx + r*cos(i*angle);
        coord[1] = cy + r*sin(i*angle);

        body.points[2*i]   = coord[0];
        body.points[2*i+1] = coord[1];
        
        body.velocities[2*i]   = 0.0f;
        body.velocities[2*i+1] = 0.0f;
    }


    return body;
}

// interp<<<blocks, threads>>>
// blocks = <ceil(pts/BlockSize) + 1, ceil (pts/BlockSize) + 1>
__global__
void interpolate_velocities_kernel(float* points, float* u_ibm, int num_points, float* u) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= num_points) return;

    float px = points[2*idx]; float py = points[2*idx + 1];
    float gx = floorf(px); float gy = floorf(py);

    float ux = 0.0f; float uy = 0.0f;
    float sum = 0.0f;

    for (int i=0; i < 2; i++) {
        for (int j=0; j < 2; j++) {
            int nx = gx + i;
            int ny = gy + j;

            if (nx >= NX || nx < 0 || ny >= NY || ny < 0)
                continue;

            float dx = px - nx; float dy = py - ny;
            float k = kernel2D(dx, dy);
            int u_idx = ny * NX + nx;

            if (idx==8) {
                printf("  Node %d: (%d, %d)\n", i*2+j, nx, ny);
                printf("    dx=%.6f, dy=%.6f\n", dx, dy);
                printf("    kernel=%.6f\n", k);
                printf("    u[%d]=(%.6f, %.6f)\n", u_idx, u[2*u_idx], u[2*u_idx+1]);
                printf("    current vel: (%.4f, %.4f)\n", ux, uy);
            }

            sum += k;
            ux += k * u[2*u_idx];
            uy += k * u[2*u_idx + 1];
        }
    }

    u_ibm[2*idx] = ux;
    u_ibm[2*idx + 1] = uy;

    

    if (fabsf(ux) > 0.1f || fabsf(uy) > 0.1f)
        printf("  Interpolated velocity for point %d (%.4f, %.4f): (%.4f, %.4f)\n\tsum=%.4f\n", idx, px, py, ux, uy, sum);
}

__global__
void compute_penalties_kernel(float* u_ibm, float* penalties, int num_points, float* rho) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= num_points) return;

    float ux = u_ibm[2*idx]; float uy = u_ibm[2*idx + 1] ;
    constexpr float u_target = 0.0f; // noslip boundaries for now
    //TODO: make this customizable if needed

    //in theory use rho=1 but for some god forsaken reason the density is decreasing on flow past cylinder,
    // likely due to zero gradient outflow. ffs
    penalties[2*idx]   = (u_target - ux);
    penalties[2*idx+1] = (u_target - uy);
    float fx = penalties[2*idx];
    float fy = penalties[2*idx + 1];

    // if (fabsf(fx) > 0.5f || fabsf(fy) > 0.5f)
    //     printf(
    //         "IBM[%d](Block %d,Thread %d): \n|"
    //         "\tVel=(%.4f,%.4f) Mag=%.4f \n| "
    //         "\tρ=%.4f \n| "
    //         "\tForce=(%.4f,%.4f) Mag=%.4f \n| "
    //         "\tError=(%.4f,%.4f) Mag=%.4f\n",
    //         idx, blockIdx.x, threadIdx.x,
    //         ux, uy, sqrtf(ux*ux + uy*uy),
    //         rho[idx],
    //         fx, fy, sqrtf(fx*fx + fy*fy),
    //         (u_target-ux), (u_target-uy), 
    //         sqrtf((u_target-ux)*(u_target-ux) + (u_target-uy)*(u_target-uy))
    //     );
}

__global__
void spread_forces_kernel(float* points, float* penalties, int num_points, float* lbm_forces) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= num_points) return;

    float px = points[2*idx]; float py = points[2*idx + 1];
    float gx = floorf(px); float gy = floorf(py);

    float fx = 0.0f; float fy = 0.0f;

    for (int i=0; i<2; i++) {
        for (int j=0; j<2; j++) {
            int nx = gx + i;
            int ny = gy + j;

            if (nx >= NX || nx < 0 || ny >= NY || ny < 0)
                continue;

            float dx = px - (gx + i); float dy = py - (gy + j);
            float k = kernel2D(dx, dy);

            fx = k * penalties[2*idx];
            fy = k * penalties[2*idx + 1];

            // if (fabsf(fx) > 0.5f || fabsf(fy) > 0.5f)
            //     printf("forces in IBM (adding): (%.4f, %.4f)\n", fx, fy);

            int node_idx = ny * NX + nx;
            atomicAdd(&lbm_forces[2*node_idx], fx);
            atomicAdd(&lbm_forces[2*node_idx+1], fy);
        }
    }

}