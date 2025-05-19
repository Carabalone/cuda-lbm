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

__global__
void interpolate_velocities_kernel(float* points, float* u_ibm, float* rho_ibm, int num_pts,
                                   float* u_lbm, float* rho_lbm) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= num_pts) return;

    float px = points[2*idx]; float py = points[2*idx + 1];
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

__global__
void compute_lagrangian_kernel(float* points, float* u_ibm, float* forces_lagrangian, int num_pts, float* rho_ibm) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= num_pts) return;

    float ux = u_ibm[get_lag_vec_index(idx, 0, num_pts)]; float uy = u_ibm[get_lag_vec_index(idx, 1, num_pts)] ;
    float px = points[2*idx]; float py = points[2*idx + 1] ;

    constexpr float u_target = 0.0f; // noslip boundaries for now
    //TODO: make this customizable if needed

    forces_lagrangian[get_lag_vec_index(idx, 0, num_pts)] = 2.0f * rho_ibm[idx] * (u_target - ux);
    forces_lagrangian[get_lag_vec_index(idx, 1, num_pts)] = 2.0f * rho_ibm[idx] * (u_target - uy);
}

__global__
void spread_forces_kernel(float* points, float* forces_lagrangian, int num_pts, float* forces_eulerian) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= num_pts) return;

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

__global__
void correct_velocities_kernel(float* u, float* f_iter, float* rho) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= NX || y >= NY) return;
    
    int node = y * NX + x;
    
    u[get_vec_index(node, 0)] = u[get_vec_index(node, 0)] + f_iter[get_vec_index(node, 0)]   / (2.0f * rho[node]);
    u[get_vec_index(node, 1)] = u[get_vec_index(node, 1)] + f_iter[get_vec_index(node, 1)] / (2.0f * rho[node]);
}

__global__
void accumulate_forces_kernel(float* forces_total, float* iter_force) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int node = y * NX + x;
    forces_total[get_vec_index(node, 0)] += iter_force[get_vec_index(node, 0)];
    forces_total[get_vec_index(node, 1)] += iter_force[get_vec_index(node, 1)];

    if (fabsf(forces_total[2 * node]) > 0.001f) {
            // printf("Node (%d, %d) | Force X: %.6f | Force Y: %.6f\n", 
            //     x, y, forces_total[2 * node], forces_total[2 * node + 1]);
    }
}