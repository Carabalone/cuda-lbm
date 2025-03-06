#include "lbm.cuh"
#include "lbm_constants.cuh"
#include "functors/includes.cuh"


#define DEBUG_KERNEL 0
#define DEBUG_NODE 5
#define VALUE_THRESHOLD 5.0f
// #define PERIODIC
// #define PERIODIC_X
// #define PERIODIC_Y

#if DEBUG_KERNEL
  #define DPRINTF(fmt, ...) printf(fmt, __VA_ARGS__)
#else
  #define DPRINTF(fmt, ...)
#endif

#ifdef D2Q9
    __constant__ float WEIGHTS[quadratures];
    __constant__ int C[quadratures * dimensions];
    __constant__ float vis;
    __constant__ float tau;
    __constant__ float omega;
    __constant__ int OPP[quadratures];
#endif

// u -> velocity, rho -> density
__device__
void LBM::equilibrium_node(float* f_eq, float ux, float uy, float rho, int node) {
    float u_dot_u = ux * ux + uy * uy;
    float cs  = 1.0f / sqrt(3.0f);
    float cs2 = cs * cs;
    float cs4 = cs2 * cs2;

    f_eq[get_node_index(node, 0)] = WEIGHTS[0]*rho*
        (1 + 0.5*pow(C[0]*ux + C[1]*uy, 2)/cs4 - 0.5*u_dot_u/cs2 + 1.0*(C[0]*ux + C[1]*uy)/cs2);

    f_eq[get_node_index(node, 1)] = WEIGHTS[1]*rho*
        (1 + 0.5*pow(C[2]*ux + C[3]*uy, 2)/cs4 - 0.5*u_dot_u/cs2 + 1.0*(C[2]*ux + C[3]*uy)/cs2);

    f_eq[get_node_index(node, 2)] = WEIGHTS[2]*rho*
        (1 + 0.5*pow(C[4]*ux + C[5]*uy, 2)/cs4 - 0.5*u_dot_u/cs2 + 1.0*(C[4]*ux + C[5]*uy)/cs2);

    f_eq[get_node_index(node, 3)] = WEIGHTS[3]*rho*
        (1 + 0.5*pow(C[6]*ux + C[7]*uy, 2)/cs4 - 0.5*u_dot_u/cs2 + 1.0*(C[6]*ux + C[7]*uy)/cs2);

    f_eq[get_node_index(node, 4)] = WEIGHTS[4]*rho*
        (1 + 0.5*pow(C[8]*ux + C[9]*uy, 2)/cs4 - 0.5*u_dot_u/cs2 + 1.0*(C[8]*ux + C[9]*uy)/cs2);

    f_eq[get_node_index(node, 5)] = WEIGHTS[5]*rho*
        (1 + 0.5*pow(C[10]*ux + C[11]*uy, 2)/cs4 - 0.5*u_dot_u/cs2 + 1.0*(C[10]*ux + C[11]*uy)/cs2);

    f_eq[get_node_index(node, 6)] = WEIGHTS[6]*rho*
        (1 + 0.5*pow(C[12]*ux + C[13]*uy, 2)/cs4 - 0.5*u_dot_u/cs2 + 1.0*(C[12]*ux + C[13]*uy)/cs2);

    f_eq[get_node_index(node, 7)] = WEIGHTS[7]*rho*
        (1 + 0.5*pow(C[14]*ux + C[15]*uy, 2)/cs4 - 0.5*u_dot_u/cs2 + 1.0*(C[14]*ux + C[15]*uy)/cs2);

    f_eq[get_node_index(node, 8)] = WEIGHTS[8]*rho*
        (1 + 0.5*pow(C[16]*ux + C[17]*uy, 2)/cs4 - 0.5*u_dot_u/cs2 + 1.0*(C[16]*ux + C[17]*uy)/cs2);

}

__global__ void equilibrium_kernel(float* f_eq, const float* rho, const float* u) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int node = y * NX + x;

    float node_rho = rho[node];
    float node_ux  = u[2 * node];
    float node_uy  = u[2 * node + 1];

    if (node == DEBUG_NODE) {
        DPRINTF("[equilibrium_kernel] Node %d (x=%d,y=%d): rho=%f, u=(%f,%f)\n",
                node, x, y, node_rho, node_ux, node_uy);
    }

    LBM::equilibrium_node(f_eq, node_ux, node_uy, node_rho, node);

    if (node == DEBUG_NODE) {
        DPRINTF("[equilibrium_kernel] f_eq[%d] = %f\n", get_node_index(node, 0), f_eq[get_node_index(node, 0)]);
    }
}

void LBM::compute_equilibrium() {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (NY + BLOCK_SIZE - 1) / BLOCK_SIZE);

    equilibrium_kernel<<<blocks, threads>>>(d_f_eq, d_rho, d_u);
    checkCudaErrors(cudaDeviceSynchronize());
}

// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// ---------------------------------------INIT----------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------

__device__
void LBM::init_node(float* f, float* f_back, float* f_eq, float* rho, float* u, int node) {

    equilibrium_node(f_eq, u[2*node], u[2*node+1], rho[node], node);

    for (int q = 0; q < quadratures; q++) {
        int i = get_node_index(node, q);
        f[i] = f_eq[i];
        f_back[i] = f_eq[i];
    }

}

// Templated kernels and templated host wrapper funcs are defined in lbm_impl.cuh
// The reason is the compiler cannot tell for what types it needs to implement the function
// if the function is defined here. We need to define it in a header file to be able to do so.
// 
// Maybe I should just move this entire thing besides send_consts to lbm_impl.cuh. I might do this
// later.

// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// ------------------------------------MACROSCOPICS-----------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------

__global__ void macroscopics_kernel(float* f, float* rho, float* u, float* force) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int node = y * NX + x;
    LBM::macroscopics_node(f, rho, u, force, node);
    if (node == DEBUG_NODE) {
        DPRINTF("[macroscopics_kernel] Node %d: rho=%f, u=(%f, %f)\n",
                node, rho[node], u[2*node], u[2*node+1]);
    }
}

void LBM::macroscopics() {
    dim3 blocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (NY+BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    macroscopics_kernel<<<blocks, threads>>>(d_f, d_rho, d_u, d_force);
    checkCudaErrors(cudaDeviceSynchronize());
}

// assumes no forcing scheme, else will have to correct velocity with forcing term
__device__
void LBM::macroscopics_node(float* f, float* rho, float* u, float* force, int node) {
    rho[node]       = 0.0f;
    u[2 * node]     = 0.0f;
    u[2 * node + 1] = 0.0f;

    for (int i=0; i < quadratures; i++) { 
         // f[get_node_index(node, i)] = 
         rho[node]       += f[get_node_index(node, i)];
         u[2 * node]     += f[get_node_index(node, i)] * C[2 * i];
         u[2 * node + 1] += f[get_node_index(node, i)] * C[2 * i + 1];
    }

    u[2 * node]     += 0.5f * force[2 * node];
    u[2 * node + 1] += 0.5f * force[2 * node + 1];

    // printf("adding %f\n", (force[2 * node]));

    u[2 * node]     *= 1.0f / rho[node];
    u[2 * node + 1] *= 1.0f / rho[node];

    // if (node == 0) {
    //     printf("Node 0: rho=%.4f, ux=%.4f, uy=%.4f\n", 
    //           rho[node], u[2*node], u[2*node+1]);
    // }
}

// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// --------------------------------------STREAMING------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------

__device__
void LBM::stream_node(float* f, float* f_back, int node) {
    const int x = node % NX;
    const int y = node / NX;
    const int baseIdx = get_node_index(node, 0);

    for (int i=1; i < quadratures; i++) {
        int x_neigh = x + C[2*i];
        int y_neigh = y + C[2*i+1];


#ifndef PERIODIC

        if (x_neigh < 0 || x_neigh >= NX || y_neigh < 0 || y_neigh >= NY)
            continue;

#else
    #ifdef PERIODIC_X
        if (y_neigh < 0 || y_neigh >= NY)
            continue;
        x_neigh = (x_neigh + NX) % NX;
    #endif
    #ifdef PERIODIC_Y
        if (x_neigh < 0 || x_neigh >= NX)
            continue;
        y_neigh = (y_neigh + NY) % NY;
    #endif
#endif

        const int idx_neigh = get_node_index(NX * y_neigh + x_neigh, i);
        float source_val = f[baseIdx + i];

        f_back[idx_neigh] = f[baseIdx + i];

        if (fabsf(f_back[idx_neigh]) > VALUE_THRESHOLD || f_back[idx_neigh] < -0.01f) {
            printf("[WARNING][stream_node] Pushing negative/large value: "
                "Node (x=%3d, y=%3d) is pushing f[%d]=% .6f in Dir %d to neighbor at (x=%3d, y=%3d)\n",
                x, y, i, f[baseIdx + i], i, x_neigh, y_neigh);
        }
    }
}

__global__ void stream_kernel(float* f, float* f_back) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int node = y * NX + x;

    if (node == DEBUG_NODE) {
        DPRINTF("[stream_kernel] Before streaming (node %d):\n", node);
        for (int i = 0; i < quadratures; i++) {
            DPRINTF("    f[%d] = %f\n", get_node_index(node, i), f[get_node_index(node, i)]);
        }
    }

    LBM::stream_node(f, f_back, node);

    if (node == DEBUG_NODE) {
        DPRINTF("[stream_kernel] After streaming (node %d):\n", node);
        for (int i = 0; i < quadratures; i++) {
            DPRINTF("    f_back[%d] = %f\n", get_node_index(node, i), f_back[get_node_index(node, i)]);
        }
    }
}

void LBM::stream() {
    dim3 blocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (NY+BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    stream_kernel<<<blocks, threads>>>(d_f, d_f_back);
    checkCudaErrors(cudaDeviceSynchronize());

    // float* temp;
    // temp = d_f;
    // d_f = d_f_back;
    // d_f_back = temp;
}

// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// ----------------------------------------COLLISION----------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------

// BGK Collision
__device__
void LBM::collide_node(float* f, float* f_back, float* f_eq, float* force, float* u, int node) {

    int node_x = node % NX;
    int node_y = node / NX;
     for (int i = 0; i < quadratures; i++) {
        int idx = get_node_index(node, i);
        float old_val = f[idx];

        // f[idx] = f[idx] * (1-omega) + omega * f_eq[idx];
        
        // Guo forcing term
        float cx  = C[2*i];
        float cy  = C[2*i+1];
        float fx  = force[2*node];
        // printf("fx: %f\n", fx);
        float fy  = force[2*node+1];
        float cs2 = 1.0f / 3.0f;
        
        float force_term = WEIGHTS[i] * (
            (1.0f - 0.5f * omega) * (
                (cx - u[2*node]) / cs2 + 
                (cx * u[2*node] + cy * u[2*node+1]) * cx / (cs2 * cs2)
            ) * fx +
            (1.0f - 0.5f * omega) * (
                (cy - u[2*node+1]) / cs2 + 
                (cx * u[2*node] + cy * u[2*node+1]) * cy / (cs2 * cs2)
            ) * fy
        );

        // printf("force_term: %f\n", force_term);
        
        float new_val = old_val - omega * (old_val - f_eq[idx]) + force_term;
        f[idx] = new_val;

        // if (node == 0) {
        //     printf("Node %d, Dir %d: f_old = %.4f, f_eq = %.4f, f_new = %.4f\n",
        //            node, i, f_old, f_eq[idx], f[idx]);
            // }

        if (fabsf(f[idx]) > VALUE_THRESHOLD || f[idx] < -0.01f) {
            printf("[WARNING][collide_node] Node %d (x=%d,y=%d), Dir %d, idx=%d: f[%d] = %f - %f*(%f - %f) = %f\n",
                   node, node_x, node_y, i, idx,
                   idx, old_val, omega, old_val, f_eq[idx], new_val);
        }
    }
}

__global__ void collide_kernel(float* f, float* f_back, float* f_eq, float* force, float* u) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int node = y * NX + x;

    if (node == DEBUG_NODE) {
        DPRINTF("[collide_kernel] Before collision (node %d):\n", node);
        for (int i = 0; i < quadratures; i++) {
            int idx = get_node_index(node, i);
            DPRINTF("    Dir %d: f[%d] = %f, f_eq = %f\n", i, idx, f[idx], f_eq[idx]);
        }
    }

    LBM::collide_node(f, f_back, f_eq, force, u, node);

    if (node == DEBUG_NODE) {
        DPRINTF("[collide_kernel] After collision (node %d):\n", node);
        for (int i = 0; i < quadratures; i++) {
            int idx = get_node_index(node, i);
            DPRINTF("    Dir %d: f[%d] = %f\n", i, idx, f[idx]);
        }
    }
}

void LBM::collide() {
    dim3 blocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (NY+BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    collide_kernel<<<blocks, threads>>>(d_f, d_f_back, d_f_eq, d_force, d_u);
    checkCudaErrors(cudaDeviceSynchronize());
}

// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// --------------------------------------BOUNDARIES-----------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------