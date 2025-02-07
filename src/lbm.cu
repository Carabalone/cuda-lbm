#include "lbm.cuh"

#ifdef D2Q9
    // look, I know this is bad. I have to define __constant__ variables in .cu files as far as I know
    // and this is the best way I can do it now.

    __constant__ float WEIGHTS[quadratures];
    __constant__ int C[quadratures * dimensions];

#endif

// u -> velocity, rho -> density
void LBM::equilibrium(float* f_eq, float ux, float uy, float rho, int node) {
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

void LBM::init_node(float* f, float* f_back, float* f_eq, float* rho, float* u, int node) {
    rho[node] = 1.0f;

    u[2 * node]     = 0.0f;
    u[2 * node + 1] = 0.0f;

    equilibrium(f_eq, u[2*node], u[2*node+1], rho[node], node);

    for (int q = 0; q < quadratures; q++) {
        int i = get_node_index(node, q);
        f[i] = f_eq[i];
        f_back[i] = f_eq[i];
    }

    // Debug
    if (node == 0) {
        for (int i = 0; i < quadratures; i++) {
            printf("f[%d] = %f\n", i, WEIGHTS[i]);
        }
    }

}

__global__ void init_kernel(float* f, float* f_back, float* f_eq, float* rho, float* u, int* debug_counter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int idx = y * NX + x;

    atomicAdd(debug_counter, 1);
    LBM::init_node(f, f_back, f_eq, rho, u, idx);
}

void LBM::init() {
    dim3 blocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE, (NY+BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    checkCudaErrors(cudaMemcpyToSymbol(WEIGHTS, h_weights, sizeof(float) * quadratures));
    checkCudaErrors(cudaMemcpyToSymbol(C, h_C, sizeof(int) * dimensions * quadratures));

    int* d_debug_counter;
    int h_debug_counter = 0;
    cudaMalloc(&d_debug_counter, 1 * sizeof(int));
    cudaMemcpy(d_debug_counter, &h_debug_counter, 1 * sizeof(int), cudaMemcpyHostToDevice);

    init_kernel<<<blocks, threads>>>(d_f, d_f_back, d_f_eq, d_rho, d_u, d_debug_counter);
    checkCudaErrors(cudaDeviceSynchronize());

    cudaMemcpy(&h_debug_counter, d_debug_counter, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_debug_counter);

    printf("[init_kernel]: Threads executed: %d\n", h_debug_counter);
}
