#ifndef LBM_STREAMING_H
#define LBM_STREAMING_H

#include "core/lbm.cuh"
#include "core/lbm_constants.cuh"
#include "functors/includes.cuh"

#define PERIODIC
#define PERIODIC_X
#define PERIODIC_Y
#define PERIODIC_Z

template <int dim>
__global__ inline void stream_kernel(float* __restrict__ f, float* __restrict__ f_back) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // NZ defaults as 1, z as 0
    if (x >= NX || y >= NY || z >= NZ) return;

    int node;

    if constexpr (dim == 2)
        node = y * NX + x;
    else
        node = (z * NX * NY + y * NX + x);

    if (node == DEBUG_NODE) {
        DPRINTF("[stream_kernel] Before streaming (node %d):\n", node);
        for (int i = 0; i < quadratures; i++) {
            DPRINTF("    f[%d] = %f\n", get_node_index(node, i), f[get_node_index(node, i)]);
        }
    }

    LBM<dim>::stream_node(f, f_back, node);

    if (node == DEBUG_NODE) {
        DPRINTF("[stream_kernel] After streaming (node %d):\n", node);
        for (int i = 0; i < quadratures; i++) {
            DPRINTF("    f_back[%d] = %f\n", get_node_index(node, i), f_back[get_node_index(node, i)]);
        }
    }
}

template <int dim>
__global__ inline void stream_and_uncorrected_macroscopics_kernel(const float* __restrict__ f,
                                                                   float* __restrict__ f_back,
                                                                   float* __restrict__ rho,
                                                                   float* __restrict__ u) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= NX || y >= NY || z >= NZ) return;

    int node;
    if constexpr (dim == 2) {
        node = y * NX + x;
    } else {
        node = z * NX * NY + y * NX + x;
    }

    float rho_local = 0.0f;
    float ux = 0.0f;
    float uy = 0.0f;
    float uz = 0.0f;

    const int idx0 = get_node_index(node, 0);
    const float f0 = f[idx0];
    f_back[idx0] = f0;
    rho_local += f0;

    for (int i = 1; i < quadratures; i++) {
        int x_src;
        int y_src;
        int z_src = 0;

        if constexpr (dim == 2) {
            x_src = x - C[2 * i];
            y_src = y - C[2 * i + 1];
        } else {
            x_src = x - C[3 * i];
            y_src = y - C[3 * i + 1];
            z_src = z - C[3 * i + 2];
        }

#ifdef PERIODIC_X
        x_src = (x_src + NX) % NX;
#endif
#ifdef PERIODIC_Y
        y_src = (y_src + NY) % NY;
#endif
#ifdef PERIODIC_Z
        if constexpr (dim == 3) {
            z_src = (z_src + NZ) % NZ;
        }
#endif

        if (x_src < 0 || x_src >= NX || y_src < 0 || y_src >= NY) {
            continue;
        }
        if constexpr (dim == 3) {
            if (z_src < 0 || z_src >= NZ) {
                continue;
            }
        }

        int node_src;
        if constexpr (dim == 2) {
            node_src = get_node_from_coords(x_src, y_src);
        } else {
            node_src = get_node_from_coords(x_src, y_src, z_src);
        }

        const float f_i = f[get_node_index(node_src, i)];
        f_back[get_node_index(node, i)] = f_i;

        rho_local += f_i;
        if constexpr (dim == 2) {
            ux += f_i * C[2 * i];
            uy += f_i * C[2 * i + 1];
        } else {
            ux += f_i * C[3 * i];
            uy += f_i * C[3 * i + 1];
            uz += f_i * C[3 * i + 2];
        }
    }

    const float inv_rho = 1.0f / rho_local;
    rho[node] = rho_local;
    u[get_vec_index(node, 0)] = ux * inv_rho;
    u[get_vec_index(node, 1)] = uy * inv_rho;
    if constexpr (dim == 3) {
        u[get_vec_index(node, 2)] = uz * inv_rho;
    }
}

template <int dim>
void LBM<dim>::stream() {
    dim3 threads, blocks;

    if constexpr (dim == 2) {
        threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
        blocks  = dim3((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                      (NY + BLOCK_SIZE - 1) / BLOCK_SIZE);
    }
    else {
        threads = dim3(32, 4, 2);
        blocks  = dim3((NX + threads.x - 1) / threads.x,
                      (NY + threads.y - 1) / threads.y,
                      (NZ + threads.z - 1) / threads.z);
    }

    stream_kernel<dim><<<blocks, threads>>>(d_f, d_f_back);
    sync_or_check_kernel();
}

template <int dim>
void LBM<dim>::stream_and_uncorrected_macroscopics() {
    dim3 threads, blocks;

    if constexpr (dim == 2) {
        threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
        blocks  = dim3((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                      (NY + BLOCK_SIZE - 1) / BLOCK_SIZE);
    }
    else {
        threads = dim3(32, 4, 2);
        blocks  = dim3((NX + threads.x - 1) / threads.x,
                      (NY + threads.y - 1) / threads.y,
                      (NZ + threads.z - 1) / threads.z);
    }

    stream_and_uncorrected_macroscopics_kernel<dim><<<blocks, threads>>>(d_f, d_f_back, d_rho, d_u);
    sync_or_check_kernel();
}

#endif // ! LBM_STREAMING_H
