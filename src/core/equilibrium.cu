#include "core/lbm.cuh"
#include "core/lbm_constants.cuh"
#include "functors/includes.cuh"

// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -------------------------------------EQUILIBRIUM-----------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------

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
