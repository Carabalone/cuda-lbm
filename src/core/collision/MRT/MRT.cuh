#ifndef LBM_COLLISION_MRT_H
#define LBM_COLLISION_MRT_H

#include "core/lbm.cuh"
#include "core/lbm_constants.cuh"

template <int dim>
struct MRT {
    __device__ __forceinline__ static
    void apply(float* f, float* f_eq, float* u, float* force, int node);

    __device__ static
    void compute_forcing_term(float* F, float* u, float* force, int node);
};


template<int dim>
__device__ __forceinline__
void MRT<dim>::apply(float* f, float* f_eq, float* u, float* force, int node) {
    float m[quadratures];
    float m_eq[quadratures];
    float m_post[quadratures];
    float old_f[quadratures]; // for debug
    

    float F[quadratures] = {0.0f};
    
    // compute_forcing_term(F, u, force, node);

    for (int k = 0; k < quadratures; k++) {
        old_f[k] = f[get_node_index(node, k)];
        
        m[k] = 0.0f;
        m_eq[k] = 0.0f;
        for (int i = 0; i < quadratures; i++) {
            m[k]    += M[k*quadratures + i] * f[get_node_index(node, i)];
            m_eq[k] += M[k*quadratures + i] * f_eq[get_node_index(node, i)];
        }
        
        float source_term = (1.0f - S[k]/2.0f) * F[k];
        m_post[k] = m[k] - S[k] * (m[k] - m_eq[k]) + source_term;

        if (node == DEBUG_NODE) {
            DPRINTF("[MRT::apply] Node %d: Moment %d: m=%f, m_eq=%f, F=%f, post=%f\n", 
                  node, k, m[k], m_eq[k], F[k], m_post[k]);
        }
    }

    for (int k = 0; k < quadratures; k++) {
        f[get_node_index(node, k)] = 0.0f;
        for (int i = 0; i < quadratures; i++) {
            f[get_node_index(node, k)] += M_inv[k*quadratures + i] * m_post[i];
        }

        float new_val = old_f[k] - 1.0f * (old_f[k] - f_eq[get_node_index(node, k)]);
        
        if (fabsf(f[get_node_index(node, k)]) > VALUE_THRESHOLD || f[get_node_index(node, k)] < -0.01f) {
            int x, y, z;
            get_coords_from_node(node, x, y, z);
            if (node == DEBUG_NODE)
                printf("[WARNING][MRT::apply] Node %d (x=%d,y=%d,z=%d), Dir %d: f[%d] = %f → %f\n",
                    node, x, y, z, k, get_node_index(node, k), old_f[k], f[get_node_index(node, k)]);

            
        }
        
        if (node == DEBUG_NODE) {
            DPRINTF("[MRT::apply] Node %d: Dir %d: %f → %f\n", 
                  node, k, old_f[k], f[get_node_index(node, k)]);
        }
    }
}


#endif // ! LBM_COLLISION_MRT_H