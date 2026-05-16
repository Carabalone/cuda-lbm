#include "UnoptimizedMRT.cuh"

template<>
__device__
void UnoptimizedMRT<3>::apply(float* f, float* f_eq, float* u, float* force, int node) {
    float m[quadratures];
    float m_eq[quadratures];
    float m_post[quadratures];
    float F[quadratures] = {0.0f};

    MRT<3>::compute_forcing_term(F, u, force, node);

    for (int k = 0; k < quadratures; k++) {
        m[k] = 0.0f;
        m_eq[k] = 0.0f;
        for (int i = 0; i < quadratures; i++) {
            const int fi = get_node_index(node, i);
            m[k]    += M[k * quadratures + i] * f[fi];
            m_eq[k] += M[k * quadratures + i] * f_eq[fi];
        }

        const float source_term = (1.0f - S[k] / 2.0f) * F[k];
        m_post[k] = m[k] - S[k] * (m[k] - m_eq[k]) + source_term;
    }

    for (int k = 0; k < quadratures; k++) {
        int f_idx = get_node_index(node, k);
        f[f_idx] = 0.0f;
        for (int i = 0; i < quadratures; i++) {
            f[f_idx] += M_inv[k * quadratures + i] * m_post[i];
        }

        if (fabsf(f[f_idx]) > VALUE_THRESHOLD || f[f_idx] < -0.01f) {
            if (node == DEBUG_NODE) {
                int x, y, z_coord;
                get_coords_from_node(node, x, y, z_coord);
                printf("[WARNING][UnoptimizedMRT<3>::apply] Node %d (x=%d,y=%d,z=%d), Dir %d: f=%f\n",
                       node, x, y, z_coord, k, f[f_idx]);
            }
        }
    }
}
