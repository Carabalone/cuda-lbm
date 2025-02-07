#include "lbm.cuh"

#ifdef D2Q9
    // look, I know this is bad. I have to define __constant__ variables in .cu files as far as I know
    // and this is the best way I can do it now.

    __constant__ float WEIGHTS[quadratures];
    __constant__ int C[quadratures * dimensions];

#endif

// u -> velocity, rho -> density
void LBM::equilibrium(float* f_eq, float ux, float uy, float rho, int node) {
    int i = 0;
    f_eq[get_node_index(node, i)]    =  rho * (ux * C[0]   +  uy * C[1]   +  1) * WEIGHTS[0];
    f_eq[get_node_index(node, i+1)]  =  rho * (ux * C[2]   +  uy * C[3]   +  1) * WEIGHTS[1];
    f_eq[get_node_index(node, i+2)]  =  rho * (ux * C[4]   +  uy * C[5]   +  1) * WEIGHTS[2];
    f_eq[get_node_index(node, i+3)]  =  rho * (ux * C[6]   +  uy * C[7]   +  1) * WEIGHTS[3];
    f_eq[get_node_index(node, i+4)]  =  rho * (ux * C[8]   +  uy * C[9]   +  1) * WEIGHTS[4];
    f_eq[get_node_index(node, i+5)]  =  rho * (ux * C[10]  +  uy * C[11]  +  1) * WEIGHTS[5];
    f_eq[get_node_index(node, i+6)]  =  rho * (ux * C[12]  +  uy * C[13]  +  1) * WEIGHTS[6];
    f_eq[get_node_index(node, i+7)]  =  rho * (ux * C[14]  +  uy * C[15]  +  1) * WEIGHTS[7];
    f_eq[get_node_index(node, i+8)]  =  rho * (ux * C[16]  +  uy * C[17]  +  1) * WEIGHTS[8];
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
        // printf("f: %f\n", f[i]);
    }

}

__global__ void init_kernel(float* f, float* f_back, float* f_eq, float* rho, float* u, int* debug_counter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int idx = y * NX + x;

    // Debug
    if (idx == 0) {
        for (int i = 0; i < quadratures; i++) {
            printf("WEIGHTS[%d] = %f\n", i, WEIGHTS[i]);
        }
        for (int i = 0; i < quadratures * dimensions; i++) {
            printf("C[%d] = %d\n", i, C[i]);
        }
    }

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
