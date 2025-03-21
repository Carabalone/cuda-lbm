#ifndef COLLISION_OPS_H
#define COLLISION_OPS_H

#include "core/lbm_constants.cuh"
#include "core/collision/adapters.cuh"

//TODO: MAYBE CHANGE MOMENT ORDER IN MRT & CM TO BE CONSISTENT. 
//TODO: This file needs a big refactor: 1. BGK forcing term computes f_eq manually
//TODO: 2. CM forcing term code is horrible
//TODO: 3. I need a way for deciding the equilibrium order.

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

template <typename AdapterType = NoAdapter>
struct CM {
    
    __device__ __forceinline__ static
    void apply(float* f, float* f_eq, float* u, float* force, int node) {
        int idx = get_node_index(node);
        float ux = u[2*node];
        float uy = u[2*node+1];

        float Fx = force[2*node];
        float Fy = force[2*node+1];

        float rho = 0.0f;
        
        // TODO: pass in later
        for (int i = 0; i < quadratures; i++) {
            rho += f[idx + i];
        }
        
        float k[quadratures];
        float k_eq[quadratures];
        float k_post[quadratures];
        float old_f[quadratures]; // for debug
        float pi[3] = {0.0f}; // bad for 3D, but ok for now.

        float test_f[quadratures] = {0.0f};

        // bad but ok
        float T_inv[quadratures * quadratures] = {0};
        
        for (int i = 0; i < quadratures; i++) {
            old_f[i] = f[idx + i];
        }
        
        k[0] = rho;    
        k[1] = 0.0f;   
        k[2] = 0.0f;   
        k[3] = 0.0f;   
        k[4] = 0.0f;   
        k[5] = 0.0f;   
        k[6] = 0.0f;   
        k[7] = 0.0f;   
        k[8] = 0.0f;   
        
        for (int i = 0; i < quadratures; i++) {
            float c_cx = C[2*i] - ux;
            float c_cy = C[2*i+1] - uy;
            float c_cx2 = c_cx * c_cx;
            float c_cy2 = c_cy * c_cy;
            float f_i = f[idx + i];
            
            k[1] += f_i * (c_cx);
            k[2] += f_i * (c_cy);
            k[3] += f_i * (c_cx2 + c_cy2);
            k[4] += f_i * (c_cx2 - c_cy2);
            k[5] += f_i * c_cx * c_cy;
            k[6] += f_i * c_cx2 * c_cy;
            k[7] += f_i * c_cx * c_cy2;
            k[8] += f_i * c_cx2 * c_cy2;

            pi[0] += f_i * C[2*i] * C[2*i];        
            pi[1] += f_i * C[2*i] * C[2*i+1];      
            pi[2] += f_i * C[2*i+1] * C[2*i+1];    

        }

        float pi_mag = sqrtf(pi[0] * pi[0] + 2.0f * pi[1] * pi[1] + pi[2] * pi[2]);

        const float cs2 = 1.0f / 3.0f;

        k_eq[0] = rho;
        k_eq[1] = 0.0f;
        k_eq[2] = 0.0f;
        k_eq[3] = 2.0f * rho * cs2;
        k_eq[4] = 0.0f;
        k_eq[5] = 0.0f;
        // The commented ones are for the 2nd order equilibrium, I am gonna use 4th order based on:
        // Universal formulation of central-moments-based lattice Boltzmann method with external forcing for the simulation of multiphysics phenomena
        // de rosis 2019.
        // k_eq[6] = -rho * ux * ux * uy;
        // k_eq[7] = -rho * ux * uy * uy;
        k_eq[6] = 0.0f;
        k_eq[7] = 0.0f;
        // k_eq[8] = rho * cs2 * cs2 * (27 * ux * ux * uy * uy + 1);
        k_eq[8] = rho * cs2 * cs2;

        // De Rosis Universal 2019 appendix b.
        float F[quadratures] = {
            0.0f,
            Fx,
            Fy,
            0.0f,
            0.0f,
            0.0f,
            Fy * cs2,
            Fx * cs2,
            0.0f
        };

        float j_mag = sqrtf(ux*ux + uy*uy) * rho;
        // printf("u_macroscopics(%.8f, %.8f)\nu_microscopic(%.8f, %.8f)\n",
        //         ux, uy, utest[0], utest[1]);
        // printf("j_mag: %.4f: sqrt(%.4f + %.4f) * %.4f\n"
        //        "with utest: j_mag: %.4f: sqrt(%.4f + %.4f) * %.4f\n",
        //         j_mag,(ux*ux), (uy*uy), rho,
        //         sqrtf(utest[0]*utest[0] + utest[1]*utest[1]),
        //         (utest[0]*utest[0]), (utest[1]*utest[1]), rho);

        float high_order_relaxation = AdapterType::compute_higher_order_relaxation(
            rho, j_mag, pi_mag, d_moment_avg);
        
        for (int i = 0; i < quadratures; i++) {
            float relaxation_rate = AdapterType::is_higher_order(i) ? 
                                    high_order_relaxation : S[i];

            if (node == DEBUG_NODE) {
                if (AdapterType::is_higher_order(i)) {
                    printf("[ACMAdapt] Node %d: Moment %d (high-order): Using relaxation = %.6f\n", 
                        node, i, relaxation_rate);
                }
            }

            k_post[i] = k[i] - relaxation_rate * (k[i] - k_eq[i]) +  
                        (1.0f - 0.5f*relaxation_rate) * F[i];
        }
        
        cm_matrix_inverse(T_inv, ux, uy);

        for (int i = 0; i < quadratures; i++) {
            f[idx + i] = 0.0f;
            for (int j = 0; j < quadratures; j++) {
                f[idx + i] += T_inv[i*quadratures + j] * k_post[j];
            }

            if (fabsf(f[idx + i]) > VALUE_THRESHOLD || f[idx + i] < -0.01f) {
                // printf("rho: m[0]: %.4f\n", k_post[0]);
                const int node_x = node % NX;
                const int node_y = node / NX;
                
                printf("[WARNING][CM::apply] Node %d (x=%d,y=%d), Dir %d: f[%d] = %f → %f\n",
                      node, node_x, node_y, i, idx + i, old_f[i], f[idx + i]);
            }
            
            if (node == DEBUG_NODE) {
                DPRINTF("[CM::apply] Node %d: Dir %d: %f → %f\n", 
                      node, i, old_f[i], f[idx + i]);
            }
        }
        
        if (node == DEBUG_NODE) {
            DPRINTF("[CM::apply] Node %d: Before collision:\n", node);
            for (int i = 0; i < quadratures; i++) {
                DPRINTF("  Dir %d: f=%f, k=%f, k_eq=%f, k_post=%f\n", 
                        i, old_f[i], k[i], k_eq[i], k_post[i]);
            }
        }
    }

    // SymPy-Generated CUDA code for central moment matrix inverse
    __device__ __forceinline__ static
    void cm_matrix_inverse(float* M_inv, float ux, float uy) {
        // Priority subexpressions
        float ux2 = ux*ux;
        float uy2 = uy*uy;
        float uxuy = ux*uy;
        float ux3 = ux*ux*ux;
        float uy3 = uy*uy*uy;
        float ux2uy = uy*(ux*ux);
        float uxuy2 = ux*(uy*uy);
        float ux2uy2 = (ux*ux)*(uy*uy);

        // Additional common subexpressions
        float x3 = -uxuy2;
        float x4 = ux + x3;
        float x5 = -uy;
        float x6 = ux2uy + x5;

        // Matrix elements
        M_inv[0 * 9 + 0] = -ux2 + ux2uy2 - uy2 + 1.0f;
        M_inv[0 * 9 + 1] = -2.0f*ux + 2.0f*uxuy2;
        M_inv[0 * 9 + 2] = 2.0f*ux2uy + 2.0f*x5;
        M_inv[0 * 9 + 3] = 0.5f*ux2 + 0.5f*uy2 - 1.0f;
        M_inv[0 * 9 + 4] = -0.5f*ux2 + 0.5f*uy2;
        M_inv[0 * 9 + 5] = 4.0f*uxuy;
        M_inv[0 * 9 + 6] = 2.0f*uy;
        M_inv[0 * 9 + 7] = 2.0f*ux;
        M_inv[0 * 9 + 8] = 1.0f;

        M_inv[1 * 9 + 0] = 0.5f*ux + 0.5f*ux2 - 0.5f*ux2uy2 + 0.5f*x3;
        M_inv[1 * 9 + 1] = ux - 0.5f*uy2 + x3 + 0.5f;
        M_inv[1 * 9 + 2] = -ux2uy - uxuy;
        M_inv[1 * 9 + 3] = -0.25f*ux - 0.25f*ux2 - 0.25f*uy2 + 0.25f;
        M_inv[1 * 9 + 4] = 0.25f*ux + 0.25f*ux2 - 0.25f*uy2 + 0.25f;
        M_inv[1 * 9 + 5] = -2.0f*uxuy + x5;
        M_inv[1 * 9 + 6] = x5;
        M_inv[1 * 9 + 7] = -ux - 0.5f;
        M_inv[1 * 9 + 8] = -0.5f;

        M_inv[2 * 9 + 0] = -0.5f*ux2uy - 0.5f*ux2uy2 + 0.5f*uy + 0.5f*uy2;
        M_inv[2 * 9 + 1] = -uxuy + x3;
        M_inv[2 * 9 + 2] = -0.5f*ux2 - ux2uy + uy + 0.5f;
        M_inv[2 * 9 + 3] = -0.25f*ux2 - 0.25f*uy2 + 0.25f*x5 + 0.25f;
        M_inv[2 * 9 + 4] = 0.25f*ux2 - 0.25f*uy2 + 0.25f*x5 - 0.25f;
        M_inv[2 * 9 + 5] = -ux - 2.0f*uxuy;
        M_inv[2 * 9 + 6] = x5 - 0.5f;
        M_inv[2 * 9 + 7] = -ux;
        M_inv[2 * 9 + 8] = -0.5f;

        M_inv[3 * 9 + 0] = -0.5f*ux + 0.5f*ux2 - 0.5f*ux2uy2 + 0.5f*uxuy2;
        M_inv[3 * 9 + 1] = ux + 0.5f*uy2 + x3 - 0.5f;
        M_inv[3 * 9 + 2] = -ux2uy + uxuy;
        M_inv[3 * 9 + 3] = 0.25f*ux - 0.25f*ux2 - 0.25f*uy2 + 0.25f;
        M_inv[3 * 9 + 4] = -0.25f*ux + 0.25f*ux2 - 0.25f*uy2 + 0.25f;
        M_inv[3 * 9 + 5] = -2.0f*uxuy + uy;
        M_inv[3 * 9 + 6] = x5;
        M_inv[3 * 9 + 7] = 0.5f - ux;
        M_inv[3 * 9 + 8] = -0.5f;

        M_inv[4 * 9 + 0] = 0.5f*ux2uy - 0.5f*ux2uy2 + 0.5f*uy2 + 0.5f*x5;
        M_inv[4 * 9 + 1] = uxuy + x3;
        M_inv[4 * 9 + 2] = 0.5f*ux2 - ux2uy + uy - 0.5f;
        M_inv[4 * 9 + 3] = -0.25f*ux2 + 0.25f*uy - 0.25f*uy2 + 0.25f;
        M_inv[4 * 9 + 4] = 0.25f*ux2 + 0.25f*uy - 0.25f*uy2 - 0.25f;
        M_inv[4 * 9 + 5] = ux - 2.0f*uxuy;
        M_inv[4 * 9 + 6] = x5 + 0.5f;
        M_inv[4 * 9 + 7] = -ux;
        M_inv[4 * 9 + 8] = -0.5f;

        M_inv[5 * 9 + 0] = 0.25f*ux2uy + 0.25f*ux2uy2 + 0.25f*uxuy + 0.25f*uxuy2;
        M_inv[5 * 9 + 1] = 0.5f*uxuy + 0.5f*uxuy2 + 0.25f*uy + 0.25f*uy2;
        M_inv[5 * 9 + 2] = 0.25f*ux + 0.25f*ux2 + 0.5f*ux2uy + 0.5f*uxuy;
        M_inv[5 * 9 + 3] = 0.125f*ux + 0.125f*ux2 + 0.125f*uy + 0.125f*uy2;
        M_inv[5 * 9 + 4] = -0.125f*ux - 0.125f*ux2 + 0.125f*uy + 0.125f*uy2;
        M_inv[5 * 9 + 5] = 0.5f*ux + uxuy + 0.5f*uy + 0.25f;
        M_inv[5 * 9 + 6] = 0.5f*uy + 0.25f;
        M_inv[5 * 9 + 7] = 0.5f*ux + 0.25f;
        M_inv[5 * 9 + 8] = 0.25f;

        M_inv[6 * 9 + 0] = 0.25f*ux2uy + 0.25f*ux2uy2 - 0.25f*uxuy + 0.25f*x3;
        M_inv[6 * 9 + 1] = 0.5f*uxuy + 0.5f*uxuy2 - 0.25f*uy2 + 0.25f*x5;
        M_inv[6 * 9 + 2] = -0.25f*ux + 0.25f*ux2 + 0.5f*ux2uy - 0.5f*uxuy;
        M_inv[6 * 9 + 3] = -0.125f*ux + 0.125f*ux2 + 0.125f*uy + 0.125f*uy2;
        M_inv[6 * 9 + 4] = 0.125f*ux - 0.125f*ux2 + 0.125f*uy + 0.125f*uy2;
        M_inv[6 * 9 + 5] = 0.5f*ux + uxuy + 0.5f*x5 - 0.25f;
        M_inv[6 * 9 + 6] = 0.5f*uy + 0.25f;
        M_inv[6 * 9 + 7] = 0.5f*ux - 0.25f;
        M_inv[6 * 9 + 8] = 0.25f;

        M_inv[7 * 9 + 0] = -0.25f*ux2uy + 0.25f*ux2uy2 + 0.25f*uxuy + 0.25f*x3;
        M_inv[7 * 9 + 1] = -0.5f*uxuy + 0.5f*uxuy2 + 0.25f*uy - 0.25f*uy2;
        M_inv[7 * 9 + 2] = 0.25f*ux - 0.25f*ux2 + 0.5f*ux2uy - 0.5f*uxuy;
        M_inv[7 * 9 + 3] = -0.125f*ux + 0.125f*ux2 + 0.125f*uy2 + 0.125f*x5;
        M_inv[7 * 9 + 4] = 0.125f*ux - 0.125f*ux2 + 0.125f*uy2 + 0.125f*x5;
        M_inv[7 * 9 + 5] = -0.5f*ux + uxuy + 0.5f*x5 + 0.25f;
        M_inv[7 * 9 + 6] = 0.5f*uy - 0.25f;
        M_inv[7 * 9 + 7] = 0.5f*ux - 0.25f;
        M_inv[7 * 9 + 8] = 0.25f;

        M_inv[8 * 9 + 0] = -0.25f*ux2uy + 0.25f*ux2uy2 - 0.25f*uxuy + 0.25f*uxuy2;
        M_inv[8 * 9 + 1] = -0.5f*uxuy + 0.5f*uxuy2 + 0.25f*uy2 + 0.25f*x5;
        M_inv[8 * 9 + 2] = -0.25f*ux - 0.25f*ux2 + 0.5f*ux2uy + 0.5f*uxuy;
        M_inv[8 * 9 + 3] = 0.125f*ux + 0.125f*ux2 + 0.125f*uy2 + 0.125f*x5;
        M_inv[8 * 9 + 4] = -0.125f*ux - 0.125f*ux2 + 0.125f*uy2 + 0.125f*x5;
        M_inv[8 * 9 + 5] = -0.5f*ux + uxuy + 0.5f*uy - 0.25f;
        M_inv[8 * 9 + 6] = 0.5f*uy - 0.25f;
        M_inv[8 * 9 + 7] = 0.5f*ux + 0.25f;
        M_inv[8 * 9 + 8] = 0.25f;

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