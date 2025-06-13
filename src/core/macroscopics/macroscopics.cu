#include "core/lbm_constants.cuh"
#include "core/macroscopics/macroscopics.cuh"

template<>
__global__ void uncorrected_macroscopics_kernel<2>(float* f, float* rho, float* u, float* force, float* pi_mag) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int node = get_node_from_coords(x, y);

    rho[node] = 0.0f;
    u[get_vec_index(node, 0)] = 0.0f;
    u[get_vec_index(node, 1)] = 0.0f;

    float pi[3] = {0.0f}; // pi_xx, pi_xy=pi_yx, pi_yy

    for (int i=0; i < quadratures; i++) { 
        float f_i = f[get_node_index(node, i)];
        rho[node] += f_i;
        u[get_vec_index(node, 0)] += f_i * C[2*i];
        u[get_vec_index(node, 1)] += f_i * C[2*i+1];
        pi[0] += f_i * C[2*i] * C[2*i];        // pi_xx
        pi[1] += f_i * C[2*i] * C[2*i+1];      // pi_xy
        pi[2] += f_i * C[2*i+1] * C[2*i+1];    // pi_yy
    }

    u[get_vec_index(node, 0)] *= 1.0f / rho[node];
    u[get_vec_index(node, 1)] *= 1.0f / rho[node];

    pi_mag[node] = sqrtf(pi[0] * pi[0] + 2.0f * pi[1] * pi[1] + pi[2] * pi[2]);

    if (node == DEBUG_NODE) {
        DPRINTF("[macroscopics_kernel] Node %d: rho=%f, u=(%f, %f)\n",
                node, rho[node], u[get_vec_index(node, 0)], u[get_vec_index(node, 1)]);
    }
}

template<>
__global__ void uncorrected_macroscopics_kernel<3>(float* f, float* rho, float* u, float* force, float* pi_mag) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= NX || y >= NY || z >= NZ) return;

    int node = get_node_from_coords(x, y, z);

    rho[node] = 0.0f;
    u[get_vec_index(node, 0)] = 0.0f;
    u[get_vec_index(node, 1)] = 0.0f;
    u[get_vec_index(node, 2)] = 0.0f;

    // pi_xx, pi_xy, pi_xz, pi_yy, pi_yz, pi_zz
    float pi[6] = {0.0f};

    for (int i=0; i < quadratures; i++) { 
        float f_i = f[get_node_index(node, i)];
        rho[node] += f_i;
        u[get_vec_index(node, 0)] += f_i * C[3*i];
        u[get_vec_index(node, 1)] += f_i * C[3*i+1];
        u[get_vec_index(node, 2)] += f_i * C[3*i+2];
        
        pi[0] += f_i * C[3*i] * C[3*i];         // pi_xx
        pi[1] += f_i * C[3*i] * C[3*i+1];       // pi_xy
        pi[2] += f_i * C[3*i] * C[3*i+2];       // pi_xz
        pi[3] += f_i * C[3*i+1] * C[3*i+1];     // pi_yy
        pi[4] += f_i * C[3*i+1] * C[3*i+2];     // pi_yz
        pi[5] += f_i * C[3*i+2] * C[3*i+2];     // pi_zz
    }

    u[get_vec_index(node, 0)] *= 1.0f / rho[node];
    u[get_vec_index(node, 1)] *= 1.0f / rho[node];
    u[get_vec_index(node, 2)] *= 1.0f / rho[node];

    pi_mag[node] = sqrtf(
        pi[0]*pi[0] + 2.0f*pi[1]*pi[1] + 2.0f*pi[2]*pi[2] + 
        pi[3]*pi[3] + 2.0f*pi[4]*pi[4] + pi[5]*pi[5]
    );

    // if (fabsf(u[get_vec_index(node, 0)] + u[get_vec_index(node, 0)] + u[get_vec_index(node, 0)]) > 1e-3)
    //     printf("[MACROSCOPICS] velocites @ node (%d, %d, %d): (%.4f, %.4f, %.4f)\n",
    //                x, y, z,
    //                u[get_vec_index(node, 0)],
    //                u[get_vec_index(node, 1)],
    //                u[get_vec_index(node, 2)]
    //           );
        

    if (node == DEBUG_NODE) {
        DPRINTF("[macroscopics_kernel] Node %d: rho=%f, u=(%f, %f, %f)\n",
                node, rho[node], u[get_vec_index(node, 0)], 
                u[get_vec_index(node, 1)], u[get_vec_index(node, 2)]);
    }
}

template<>
__global__ void correct_macroscopics_kernel<2>(float* f, float* rho, float* u, float* force, float* pi_mag) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;

    int node = get_node_from_coords(x, y);

    u[get_vec_index(node, 0)] += 0.5f * force[get_vec_index(node, 0)] / rho[node];
    u[get_vec_index(node, 1)] += 0.5f * force[get_vec_index(node, 1)] / rho[node];

}

template<>
__global__ void correct_macroscopics_kernel<3>(float* f, float* rho, float* u, float* force, float* pi_mag) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= NX || y >= NY || z >= NZ) return;

    int node = get_node_from_coords(x, y, z);

    u[get_vec_index(node, 0)] += 0.5f * force[get_vec_index(node, 0)] / rho[node];
    u[get_vec_index(node, 1)] += 0.5f * force[get_vec_index(node, 1)] / rho[node];
    u[get_vec_index(node, 2)] += 0.5f * force[get_vec_index(node, 2)] / rho[node];

    // if (node % 200 == 0)
    //     printf("u={%.4f, %.4f, %.4f}\n", u[get_vec_index(node, 0)], u[get_vec_index(node, 1)], u[get_vec_index(node, 2)]);
}
