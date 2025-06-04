#include "IBM/IBMManager.cuh"

template <>
__global__
void interpolate_velocities_kernel<2>(float* points, float* u_ibm, float* rho_ibm, int num_pts,
                                   float* u_lbm, float* rho_lbm) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= num_pts) return;

    float px = points[get_pt_index(idx, 0)]; float py = points[get_pt_index(idx, 1)];
    float gx = floorf(px); float gy = floorf(py);

    float ux = 0.0f; float uy = 0.0f;
    float rho = 0.0f;

    for (int i=0; i < 2; i++) {
        for (int j=0; j < 2; j++) {
            int nx = gx + i;
            int ny = gy + j;

            if (nx >= NX || nx < 0 || ny >= NY || ny < 0)
                continue;

            float dx = px - nx; float dy = py - ny;
            float k = kernel2D(dx, dy);
            int u_idx = nx + ny * NX;

            // if (idx==8) {
            //     printf("  Node %d: (%d, %d)\n", i*2+j, nx, ny);
            //     printf("    dx=%.6f, dy=%.6f\n", dx, dy);
            //     printf("    kernel=%.6f\n", k);
            //     printf("    u[%d]=(%.6f, %.6f)\n", u_idx, u_lbm[2*u_idx], u_lbm[2*u_idx+1]);
            //     printf("    current vel: (%.4f, %.4f)\n", ux, uy);
            // }

            rho += k * rho_lbm[u_idx];
            ux  += k * u_lbm[get_vec_index(u_idx, 0)];
            uy  += k * u_lbm[get_vec_index(u_idx, 1)];
        }
    }

    u_ibm[get_lag_vec_index(idx, 0, num_pts)] = ux;
    u_ibm[get_lag_vec_index(idx, 1, num_pts)] = uy;
    rho_ibm[idx] = rho;

    if (fabsf(ux) > 0.1f || fabsf(uy) > 0.1f)
        printf("  Interpolated velocity for point %d (%.4f, %.4f): (%.4f, %.4f)\n\trho=%.4f\n", idx, px, py, ux, uy, rho);
}

template <>
__global__
void interpolate_velocities_kernel<3>(float* points, float* u_ibm, float* rho_ibm, int num_pts,
                                   float* u_lbm, float* rho_lbm) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= num_pts) return;

    float px = points[get_pt_index(idx, 0)]; float py = points[get_pt_index(idx, 1)]; float pz = points[get_pt_index(idx, 2)];
    float gx = floorf(px); float gy = floorf(py); float gz = floorf(pz);

    float ux = 0.0f; float uy = 0.0f; float uz = 0.0f;
    float rho = 0.0f;

    for (int i=0; i < 2; i++) {
        for (int j=0; j < 2; j++) {
            for (int k=0; k < 2; k++) {
                int nx = gx + i;
                int ny = gy + j;
                int nz = gz + k;

                if (nx >= NX || nx < 0 || ny >= NY || ny < 0 || nz >= NZ || nz < 0)
                    continue;

                float dx = px - nx; float dy = py - ny; float dz = pz - nz;
                float kernel = kernel3D(dx, dy, dz);
                int u_idx = nx + ny * NX + nz * NX * NY;

                // if (idx==8) {
                //     printf("  Node %d: (%d, %d)\n", i*2+j, nx, ny);
                //     printf("    dx=%.6f, dy=%.6f\n", dx, dy);
                //     printf("    kernel=%.6f\n", k);
                //     printf("    u[%d]=(%.6f, %.6f)\n", u_idx, u_lbm[2*u_idx], u_lbm[2*u_idx+1]);
                //     printf("    current vel: (%.4f, %.4f)\n", ux, uy);
                // }

                rho += kernel * rho_lbm[u_idx];
                ux  += kernel * u_lbm[get_vec_index(u_idx, 0)];
                uy  += kernel * u_lbm[get_vec_index(u_idx, 1)];
                uz  += kernel * u_lbm[get_vec_index(u_idx, 2)];
            }
        }
    }

    u_ibm[get_lag_vec_index(idx, 0, num_pts)] = ux;
    u_ibm[get_lag_vec_index(idx, 1, num_pts)] = uy;
    u_ibm[get_lag_vec_index(idx, 2, num_pts)] = uz;
    rho_ibm[idx] = rho;

    if (fabsf(ux) > 0.12f || fabsf(uy) > 0.12f || fabsf(uz) >= 0.12f)
        printf("  Interpolated velocity for point %d (%.4f, %.4f, %.4f): (%.4f, %.4f, %.4f)\n\trho=%.4f\n",
             idx, px, py, pz, ux, uy, uz, rho);
}

template <>
__global__
void spread_forces_kernel<2>(float* points, float* forces_lagrangian, int num_pts, float* forces_eulerian) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= num_pts) return;

    float px = points[get_pt_index(idx, 0)]; float py = points[get_pt_index(idx, 1)];
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

            fx = k * forces_lagrangian[get_lag_vec_index(idx, 0, num_pts)];
            fy = k * forces_lagrangian[get_lag_vec_index(idx, 1, num_pts)];

            // if (fabsf(fx) > 0.5f || fabsf(fy) > 0.5f)
            //     printf("forces in IBM (adding): (%.4f, %.4f)\n", fx, fy);

            int node_idx = ny * NX + nx;
            atomicAdd(&forces_eulerian[get_vec_index(node_idx, 0)], fx);
            atomicAdd(&forces_eulerian[get_vec_index(node_idx, 1)], fy);
        }
    }
}

template <>
__global__
void spread_forces_kernel<3>(float* points, float* forces_lagrangian, int num_pts, float* forces_eulerian) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= num_pts) return;

    float px = points[get_pt_index(idx, 0)]; float py = points[get_pt_index(idx, 1)]; float pz = points[get_pt_index(idx, 2)];
    float gx = floorf(px); float gy = floorf(py); float gz = floorf(pz);

    float fx = 0.0f; float fy = 0.0f; float fz = 0.0f;

    for (int i=0; i<2; i++) {
        for (int j=0; j<2; j++) {
            for (int k=0; k<2; k++) {
                int nx = gx + i;
                int ny = gy + j;
                int nz = gz + k;

                if (nx >= NX || nx < 0 || ny >= NY || ny < 0 || nz >= NZ || nz < 0)
                    continue;

                float dx = px - (gx + i); float dy = py - (gy + j); float dz = pz - (gz + k);
                float kernel = kernel3D(dx, dy, dz);

                fx = kernel * forces_lagrangian[get_lag_vec_index(idx, 0, num_pts)];
                fy = kernel * forces_lagrangian[get_lag_vec_index(idx, 1, num_pts)];
                fz = kernel * forces_lagrangian[get_lag_vec_index(idx, 2, num_pts)];

                // if (fabsf(fx) > 0.5f || fabsf(fy) > 0.5f || fabsf(fz) > 0.5f)
                //     printf("forces in IBM (adding): (%.4f, %.4f, %.4f)\n", fx, fy, fz);

                int node_idx = nz * NX * NY + ny * NX + nx;
                atomicAdd(&forces_eulerian[get_vec_index(node_idx, 0)], fx);
                atomicAdd(&forces_eulerian[get_vec_index(node_idx, 1)], fy);
                atomicAdd(&forces_eulerian[get_vec_index(node_idx, 2)], fz);
            }
        }
    }
}
