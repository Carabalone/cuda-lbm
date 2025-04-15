#ifndef LBM_COLLISION_MRT_H
#define LBM_COLLISION_MRT_H

#include "core/lbm.cuh"
#include "core/lbm_constants.cuh"

template <int dim>
struct MRT {
    __device__ __forceinline__ static
    void apply(float* f, float* f_eq, float* u, float* force, int node);
};


template<>
__device__ __forceinline__
void MRT<2>::apply(float* f, float* f_eq, float* u, float* force, int node) {
    int idx = get_node_index(node);
    float m[quadratures];
    float m_eq[quadratures];
    float m_post[quadratures];
    float old_f[quadratures]; // for debug

    float fx = force[get_vec_index(node, 0)];
    float fy = force[get_vec_index(node, 1)];
    float ux = u[get_vec_index(node, 0)];
    float uy = u[get_vec_index(node, 1)];

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
            int x, y, z;
            get_coords_from_node(node, x, y, z);
            
            printf("[WARNING][MRT::apply] Node %d (x=%d,y=%d), Dir %d: f[%d] = %f → %f\n",
                  node, x, y, k, idx + k, old_f[k], f[idx + k]);
        }
        
        if (node == DEBUG_NODE) {
            DPRINTF("[MRT::apply] Node %d: Dir %d: %f → %f\n", 
                  node, k, old_f[k], f[idx + k]);
        }
    }
}

#endif // ! LBM_COLLISION_MRT_H