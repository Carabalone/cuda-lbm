#include "MRT.cuh"


template <>
void MRT<2>::compute_forcing_term(float* F, float* u, float* force, int node) {

    float fx = force[get_vec_index(node, 0)];
    float fy = force[get_vec_index(node, 1)];
    float ux = u[get_vec_index(node, 0)];
    float uy = u[get_vec_index(node, 1)];

    // Orthogonalized Guo Forcing scheme for MRT
    // G. Silva, V. Semiao, J. Fluid Mech. 698, 282 (2012)
    F[0] = 0.0f;                          // ρ (density) - conserved; no force contribution
    F[1] = 6.0f * (fx*ux + fy*uy);        // e (energy)
    F[2] = -6.0f * (fx*ux + fy*uy);       // ε (energy squared)
    F[3] = fx;                            // jx (x-momentum)
    F[4] = fy;                            // jy (y-momentum)
    F[5] = -fx;                           // qx (x heat flux)
    F[6] = -fy;                           // qy (y heat flux)
    F[7] = 2.0f * (fx*ux - fy*uy);        // pxx (xx stress)
    F[8] = fx*uy + fy*ux;                 // pxy (xy stress)

}

template<>
__device__
void MRT<2>::apply(float* f, float* f_eq, float* u, float* force, int node) {
    float m[quadratures];
    float m_eq[quadratures];
    float m_post[quadratures];
    float old_f[quadratures]; // for debug
    
    float F[quadratures] = {0.0f};
    
    compute_forcing_term(F, u, force, node);

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


template <>
void MRT<3>::compute_forcing_term(float* F, float* u, float* force, int node) {

    float fx = force[get_vec_index(node, 0)];
    float fy = force[get_vec_index(node, 1)];
    float fz = force[get_vec_index(node, 2)];

    float ux = u[get_vec_index(node, 0)];
    float uy = u[get_vec_index(node, 1)];
    float uz = u[get_vec_index(node, 2)];

    F[0] = 0.0f;
    F[1] = fx;
    F[2] = fy;
    F[3] = fz;
    F[4] = fx*uy + fy*ux;
    F[5] = fx*uz + fz*ux;
    F[6] = fy*uz + fz*uy;
    F[7] = 2.0f*fx*ux - 2.0f*fy*uy;
    F[8] = 2.0f*fx*ux - 2.0f*fz*uz;
    F[9] = 2.0f*fx*ux + 2.0f*fy*uy + 2.0f*fz*uz;
    F[10] = (2.0f/3.0f)*fx;
    F[11] = (2.0f/3.0f)*fy;
    F[12] = (2.0f/3.0f)*fz;
    F[13] = 0.0f;
    F[14] = 0.0f;
    F[15] = 0.0f;
    F[16] = 0.0f;
    F[17] = (4.0f/3.0f)*fx*ux + (4.0f/3.0f)*fy*uy + (4.0f/3.0f)*fz*uz;
    F[18] = (4.0f/3.0f)*fx*ux;
    F[19] = (2.0f/3.0f)*fy*uy - 2.0f/3.0f*fz*uz;
    F[20] = (1.0f/3.0f)*fy*uz + (1.0f/3.0f)*fz*uy;
    F[21] = (1.0f/3.0f)*fx*uz + (1.0f/3.0f)*fz*ux;
    F[22] = (1.0f/3.0f)*fx*uy + (1.0f/3.0f)*fy*ux;
    F[23] = (1.0f/9.0f)*fx;
    F[24] = (1.0f/9.0f)*fy;
    F[25] = (1.0f/9.0f)*fz;
    F[26] = (2.0f/9.0f)*fx*ux + (2.0f/9.0f)*fy*uy + (2.0f/9.0f)*fz*uz;
}


// De Rosis Universal Formulation of Central Moments (2019) Appendix E.
template<>
__device__
void MRT<3>::apply(float* f, float* f_eq, float* u, float* force, int node) {
    float m_post[quadratures];
    
    float F[quadratures] = {0.0f};
    
    compute_forcing_term(F, u, force, node);

    float cs2 = 1.0f/3.0f;
    float ux = u[get_vec_index(node, 0)];
    float uy = u[get_vec_index(node, 1)];
    float uz = u[get_vec_index(node, 2)];
    float ux2 = ux * ux;
    float uy2 = uy * uy;
    float uz2 = uz * uz;
    float rho = 0.0f;
    float m4 = 0.0f, m5 = 0.0f, m6 = 0.0f, m7 = 0.0f, m8 = 0.0f;
    
    for (int i = 0; i < quadratures; i++) {
        float f_i = f[get_node_index(node, i)];
        rho += f_i; // we can pass in as arg, but I don't want to change the apply signature for all Collision Ops.

        float cix = C[3*i];
        float ciy = C[3*i+1];
        float ciz = C[3*i+2];
        
        m4 += f_i * cix * ciy;
        m5 += f_i * cix * ciz;
        m6 += f_i * ciy * ciz;
        m7 += f_i * (cix*cix - ciy*ciy);
        m8 += f_i * (cix*cix - ciz*ciz);
    }
    
    // conserved moments (F=0)
    m_post[0] = rho;
    m_post[1] = rho * ux;
    m_post[2] = rho * uy;
    m_post[3] = rho * uz;

    // affected by relaxation rate
    m_post[4] = (1.0f - S[4]) * m4 + S[4] * rho * ux * uy + (1.0f - S[4]/2.0f) * F[4];
    m_post[5] = (1.0f - S[5]) * m5 + S[5] * rho * ux * uz + (1.0f - S[5]/2.0f) * F[5];
    m_post[6] = (1.0f - S[6]) * m6 + S[6] * rho * uy * uz + (1.0f - S[6]/2.0f) * F[6];
    m_post[7] = (1.0f - S[7]) * m7 + S[7] * rho * (ux2 - uy2) + (1.0f - S[7]/2.0f) * F[7];
    m_post[8] = (1.0f - S[8]) * m8 + S[8] * rho * (ux2 - uz2) + (1.0f - S[8]/2.0f) * F[8];
    
    // relaxation rate = 1 -> (1 - S[i]/2) = (1-1/2) = 0.5
    m_post[9] = rho * (ux2 + uy2 + uz2 + 1.0f) + 0.5f* F[9];
    m_post[10] = rho * cs2 * ux * (3.0f*uy2 + 3.0f*uz2 + 2.0f) + 0.5f * F[10];
    m_post[11] = rho * cs2 * uy * (3.0f*ux2 + 3.0f*uz2 + 2.0f) + 0.5f * F[11];
    m_post[12] = rho * cs2 * uz * (3.0f*ux2 + 3.0f*uy2 + 2.0f) + 0.5f * F[12];
    m_post[13] = rho * ux * (uy2 - uz2) + 0.5f * F[13];
    m_post[14] = rho * uy * (ux2 - uz2) + 0.5f * F[14];
    m_post[15] = rho * uz * (ux2 - uy2) + 0.5f * F[15];
    m_post[16] = rho * ux * uy * uz + 0.5f * F[16];
    m_post[17] = rho * cs2 * (3.0f*(ux2*uy2 + ux2*uz2 + uy2*uz2) + 2.0f*(ux2 + uy2 + uz2) + 1.0f) + 0.5f * F[17];
    m_post[18] = rho * cs2 * cs2 * (9.0f*(ux2*uy2 + ux2*uz2 - uy2*uz2) + 6.0f*ux2 + 1.0f) + 0.5f * F[18];
    m_post[19] = rho * cs2 * (uy2 - uz2) * (2.0f*ux2 + 1.0f) + 0.5f * F[19];
    m_post[20] = rho * cs2 * uy * uz * (3.0f*ux2 + 1.0f) + 0.5f * F[20];
    m_post[21] = rho * cs2 * ux * uz * (3.0f*uy2 + 1.0f) + 0.5f * F[21];
    m_post[22] = rho * cs2 * ux * uy * (3.0f*uz2 + 1.0f) + 0.5f * F[22];
    m_post[23] = rho * cs2 * cs2 * ux * (3.0f*uy2 + 1.0f) * (3.0f*uz2 + 1.0f) + 0.5f * F[23];
    m_post[24] = rho * cs2 * cs2 * uy * (3.0f*ux2 + 1.0f) * (3.0f*uz2 + 1.0f) + 0.5f * F[24];
    m_post[25] = rho * cs2 * cs2 * uz * (3.0f*ux2 + 1.0f) * (3.0f*uy2 + 1.0f) + 0.5f * F[25];
    m_post[26] = rho * cs2 * cs2 * cs2 * (3.0f*ux2 + 1.0f) * (3.0f*uy2 + 1.0f) * (3.0f*uz2 + 1.0f) + 0.5f * F[26];
    
    for (int k = 0; k < quadratures; k++) {
        int f_idx = get_node_index(node, k);
        f[f_idx] = 0.0f;
        for (int i = 0; i < quadratures; i++) {
            f[f_idx] += M_inv[k*quadratures + i] * m_post[i];
        }
        
        if (fabsf(f[f_idx]) > VALUE_THRESHOLD || f[f_idx] < -0.01f) {
            int x, y, z;
            get_coords_from_node(node, x, y, z);
            if (node == DEBUG_NODE)
                printf("[WARNING][MRT::apply] Node %d (x=%d,y=%d,z=%d), Dir %d: f[%d] = %f → %f\n",
                    node, x, y, z, k, f_idx, -1.0f, f[f_idx]);
        }
    }
}
