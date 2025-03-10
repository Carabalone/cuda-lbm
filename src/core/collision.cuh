#ifndef COLLISION_OPS_H
#define COLLISION_OPS_H

#include "core/lbm_constants.cuh"

struct BGK {

    __device__ __forceinline__ static
    void apply(float* f, float* f_eq, float* u, float* force, int node) {
        for (int q = 0; q < quadratures; q++) {

            int idx = get_node_index(node, q);
            float cx  = C[2*q];
            float cy  = C[2*q+1];
            float fx  = force[2*node];
            
            float fy  = force[2*node+1];
            float cs2 = 1.0f / 3.0f;
            
            float force_term = WEIGHTS[q] * (
                (1.0f - 0.5f * omega) * (
                    (cx - u[2*node]) / cs2 + 
                    (cx * u[2*node] + cy * u[2*node+1]) * cx / (cs2 * cs2)
                ) * fx +
                (1.0f - 0.5f * omega) * (
                    (cy - u[2*node+1]) / cs2 + 
                    (cx * u[2*node] + cy * u[2*node+1]) * cy / (cs2 * cs2)
                ) * fy
            );
    
            float old_val = f[idx];
            float new_val = f[idx] - omega * (f[idx] - f_eq[idx]) + force_term;
            f[idx] = new_val;
    
            if (fabsf(f[idx]) > 1.1f || f[idx] < -0.01f) {
                const int node_x = node % NX;
                const int node_y = node / NX;
    
                printf("[WARNING][collide_node] Node %d (x=%d,y=%d), Dir %d, idx=%d: f[%d] = %f - %f*(%f - %f) = %f\n",
                       node, node_x, node_y, q, idx,
                       idx, old_val, omega, old_val, f_eq[idx], new_val);
            }
        }
    }
};

struct MRT {

    __device__ __forceinline__ static
    void apply(float* f, float* f_eq, float* u, float* force, int node) {
        int idx = get_node_index(node);
        float m[quadratures];
        float m_eq[quadratures];
        float m_post[quadratures];
        float old_f[quadratures]; // for debug

        float fx = force[2*node];
        float fy = force[2*node+1];
        float ux = u[2*node];
        float uy = u[2*node+1];

        // Orthogonalized Guo Forcing scheme for MRT
        // G. Silva, V. Semiao, J. Fluid Mech. 698, 282 (2012)
        float F[quadratures] = {
            0.0f,                          // ρ (density) - conserved, no force contribution
            6.0f * (fx*ux + fy*uy),        // e (energy)
            -6.0f * (fx*ux + fy*uy),       // ε (energy squared)
            fx,                            // jx (x-momentum)
            fy,                            // jy (y-momentum)
            -fx,                           // qx (x heat flux)
            -fy,                           // qy (y heat flux)
            2.0f * (fx*ux - fy*uy),        // pxx (xx stress)
            fx*uy + fy*ux                  // pxy (xy stress)
        };

        for (int k = 0; k < quadratures; k++) {
            old_f[k] = f[idx + k];
            
            m[k] = 0.0f;
            m_eq[k] = 0.0f;
            for (int i = 0; i < quadratures; i++) {
                m[k]    += M[k*quadratures + i] * f[idx + i];
                m_eq[k] += M[k*quadratures + i] * f_eq[idx + i];
            }

            
            float source_term = (1.0f - S[k]/2.0f) * F[k];
            m_post[k] = m[k] - S[k] * (m[k] - m_eq[k]) + source_term;

            if (node == DEBUG_NODE) {
                DPRINTF("[MRT::apply] Node %d: Moment %d: m=%f, m_eq=%f, F=%f, post=%f\n", 
                      node, k, m[k], m_eq[k], F[k], m_post[k]);
            }
        }

        for (int k = 0; k < quadratures; k++) {
            f[idx + k] = 0.0f;
            for (int i = 0; i < quadratures; i++) {
                f[idx + k] += M_inv[k*quadratures + i] * m_post[i];
            }

            float new_val = old_f[k] - 1.0f * (old_f[k] - f_eq[idx + k]);

            
            if (fabsf(f[idx + k]) > VALUE_THRESHOLD || f[idx + k] < -0.01f) {

                printf("rho: m[0]: %.4f\n", m_post[0]);
                const int node_x = node % NX;
                const int node_y = node / NX;
                
                // printf("f[%d + %d] = %.4f | BGK would be: %.4f - 1.0 * (%.4f - %.4f) + %.4f = %.4f\n",
        //    idx, k, f[idx + k], old_f[k], old_f[k], f_eq[idx + k], 0.0f, new_val);
                printf("[WARNING][MRT::apply] Node %d (x=%d,y=%d), Dir %d: f[%d] = %f → %f\n",
                      node, node_x, node_y, k, idx + k, old_f[k], f[idx + k]);
            }
            
            if (node == DEBUG_NODE) {
                DPRINTF("[MRT::apply] Node %d: Dir %d: %f → %f\n", 
                      node, k, old_f[k], f[idx + k]);
            }
        }
    }
};


// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// ----------------------------------------COLLISION----------------------------------------------------
// -----------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------

template <typename CollisionOp>
__global__ void collide_kernel(float* f, float* f_eq, float* u, float* force) {
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

    CollisionOp::apply(f, f_eq, u, force, node);

    if (node == DEBUG_NODE) {
        DPRINTF("[collide_kernel] After collision (node %d):\n", node);
        for (int i = 0; i < quadratures; i++) {
            int idx = get_node_index(node, i);
            DPRINTF("    Dir %d: f[%d] = %f\n", i, idx, f[idx]);
        }
    }
}

template <typename CollisionOp>
void LBM::collide() {
    dim3 blocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (NY+BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    collide_kernel<CollisionOp><<<blocks, threads>>>(d_f, d_f_eq, d_u, d_force);
    checkCudaErrors(cudaDeviceSynchronize());
}

#endif // ! COLLISION_OPS_H