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

__device__ static
float compute_forcing_term_component(int moment_index, float ux, float uy,
                                float uz, float fx, float fy, float fz) {
    switch (moment_index) {
        case 0: return 0.0f;
        case 1: return fx;
        case 2: return fy;
        case 3: return fz;
        case 4: return fx * uy + fy * ux;
        case 5: return fx * uz + fz * ux;
        case 6: return fy * uz + fz * uy;
        case 7: return 2.0f * fx * ux - 2.0f * fy * uy;
        case 8: return 2.0f * fx * ux - 2.0f * fz * uz;
        case 9: return 2.0f * fx * ux + 2.0f * fy * uy + 2.0f * fz * uz;
        case 10: return (2.0f / 3.0f) * fx;
        case 11: return (2.0f / 3.0f) * fy;
        case 12: return (2.0f / 3.0f) * fz;
        case 13: return 0.0f;
        case 14: return 0.0f;
        case 15: return 0.0f;
        case 16: return 0.0f;
        case 17: return (4.0f/3.0f)*fx*ux + (4.0f/3.0f)*fy*uy + (4.0f/3.0f)*fz*uz;
        case 18: return (4.0f/3.0f)*fx*ux;
        case 19: return (2.0f/3.0f)*fy*uy - (2.0f/3.0f)*fz*uz;
        case 20: return (1.0f/3.0f)*fy*uz + (1.0f/3.0f)*fz*uy;
        case 21: return (1.0f/3.0f)*fx*uz + (1.0f/3.0f)*fz*ux;
        case 22: return (1.0f/3.0f)*fx*uy + (1.0f/3.0f)*fy*ux;
        case 23: return (1.0f/9.0f)*fx;
        case 24: return (1.0f/9.0f)*fy;
        case 25: return (1.0f/9.0f)*fz;
        case 26: return (2.0f/9.0f)*fx*ux + (2.0f/9.0f)*fy*uy + (2.0f/9.0f)*fz*uz;
        default:
            return -99.0f; // should not happen
    }
}

__device__ static
float compute_m_post_component(
        int moment_index, float rho, float ux, float uy, float uz,
        float m4, float m5, float m6, float m7, float m8,
        float F_i
    ) {
    float m_post_val;
    float ux2 = ux * ux;
    float uy2 = uy * uy;
    float uz2 = uz * uz;
    float cs2 = 1.0f / 3.0f;

    switch (moment_index) {
    case 0: m_post_val = rho; break;
    case 1: m_post_val = rho * ux; break;
    case 2: m_post_val = rho * uy; break;
    case 3: m_post_val = rho * uz; break;
    case 4: m_post_val = (1.0f - S[4]) * m4 + S[4] * rho * ux * uy + (1.0f - 0.5f * S[4]) * F_i; break;
    case 5: m_post_val = (1.0f - S[5]) * m5 + S[5] * rho * ux * uz + (1.0f - 0.5f * S[5]) * F_i; break;
    case 6: m_post_val = (1.0f - S[6]) * m6 + S[6] * rho * uy * uz + (1.0f - 0.5f * S[6]) * F_i; break;
    case 7: m_post_val = (1.0f - S[7]) * m7 + S[7] * rho * (ux2 - uy2) + (1.0f - 0.5f * S[7]) * F_i; break;
    case 8: m_post_val = (1.0f - S[8]) * m8 + S[8] * rho * (ux2 - uz2) + (1.0f - 0.5f * S[8]) * F_i; break;
    
    case 9:  m_post_val = rho * (ux2 + uy2 + uz2 + 1.0f) + 0.5f * F_i; break;
    case 10: m_post_val = rho * cs2 * ux * (3.0f*uy2 + 3.0f*uz2 + 2.0f) + 0.5f * F_i; break;
    case 11: m_post_val = rho * cs2 * uy * (3.0f*ux2 + 3.0f*uz2 + 2.0f) + 0.5f * F_i; break;
    case 12: m_post_val = rho * cs2 * uz * (3.0f*ux2 + 3.0f*uy2 + 2.0f) + 0.5f * F_i; break;
    case 13: m_post_val = rho * ux * (uy2 - uz2) + 0.5f * F_i; break;
    case 14: m_post_val = rho * uy * (ux2 - uz2) + 0.5f * F_i; break;
    case 15: m_post_val = rho * uz * (ux2 - uy2) + 0.5f * F_i; break;
    case 16: m_post_val = rho * ux * uy * uz + 0.5f * F_i; break;
    case 17: m_post_val = rho * cs2 * (3.0f*(ux2*uy2 + ux2*uz2 + uy2*uz2) + 2.0f*(ux2 + uy2 + uz2) + 1.0f) + 0.5f * F_i; break;
    case 18: m_post_val = rho * cs2 * cs2 * (9.0f*(ux2*uy2 + ux2*uz2 - uy2*uz2) + 6.0f*ux2 + 1.0f) + 0.5f * F_i; break;
    case 19: m_post_val = rho * cs2 * (uy2 - uz2) * (2.0f*ux2 + 1.0f) + 0.5f * F_i; break;
    case 20: m_post_val = rho * cs2 * uy * uz * (3.0f*ux2 + 1.0f) + 0.5f * F_i; break;
    case 21: m_post_val = rho * cs2 * ux * uz * (3.0f*uy2 + 1.0f) + 0.5f * F_i; break;
    case 22: m_post_val = rho * cs2 * ux * uy * (3.0f*uz2 + 1.0f) + 0.5f * F_i; break;
    case 23: m_post_val = rho * cs2 * cs2 * ux * (3.0f*uy2 + 1.0f) * (3.0f*uz2 + 1.0f) + 0.5f * F_i; break;
    case 24: m_post_val = rho * cs2 * cs2 * uy * (3.0f*ux2 + 1.0f) * (3.0f*uz2 + 1.0f) + 0.5f * F_i; break;
    case 25: m_post_val = rho * cs2 * cs2 * uz * (3.0f*ux2 + 1.0f) * (3.0f*uy2 + 1.0f) + 0.5f * F_i; break;
    case 26: m_post_val = rho * cs2 * cs2 * cs2 * (3.0f*ux2 + 1.0f) * (3.0f*uy2 + 1.0f) * (3.0f*uz2 + 1.0f) + 0.5f * F_i; break;
    default:
        m_post_val = -99.0f; // should not happen
        break;
    }
    return m_post_val;
}

__device__ static float compute_m_post_combined(
    int moment_index,
    float rho,
    float ux, float uy, float uz,
    float fx, float fy, float fz,
    float m4, float m5, float m6, float m7, float m8
) {
    float ux2 = ux * ux;
    float uy2 = uy * uy;
    float uz2 = uz * uz;
    float cs2 = 1.0f / 3.0f;

    float m_post_val;

    switch (moment_index) {
        case 0: // Mass
            // m*_0 = rho
            m_post_val = rho;
            break;
        case 1: // Momentum x
            // m*_1 = rho * ux
            m_post_val = rho * ux;
            break;
        case 2: // Momentum y
            // m*_2 = rho * uy
            m_post_val = rho * uy;
            break;
        case 3: // Momentum z
            // m*_3 = rho * uz
            m_post_val = rho * uz;
            break;
        case 4: // Viscous stress xy
            // m*_4 = (1-S_4) m_4 + S_4 m^eq_4 + (1-0.5 S_4) F_4
            // m^eq_4 = rho * ux * uy
            // F_4 = fx * uy + fy * ux
            m_post_val = (1.0f - S[4]) * m4 + S[4] * rho * ux * uy +
                         (1.0f - 0.5f * S[4]) * (fx * uy + fy * ux);
            break;
        case 5: // Viscous stress xz
            // m*_5 = (1-S_5) m_5 + S_5 m^eq_5 + (1-0.5 S_5) F_5
            // m^eq_5 = rho * ux * uz
            // F_5 = fx * uz + fz * ux
            m_post_val = (1.0f - S[5]) * m5 + S[5] * rho * ux * uz +
                         (1.0f - 0.5f * S[5]) * (fx * uz + fz * ux);
            break;
        case 6: // Viscous stress yz
            // m*_6 = (1-S_6) m_6 + S_6 m^eq_6 + (1-0.5 S_6) F_6
            // m^eq_6 = rho * uy * uz
            // F_6 = fy * uz + fz * uy
            m_post_val = (1.0f - S[6]) * m6 + S[6] * rho * uy * uz +
                         (1.0f - 0.5f * S[6]) * (fy * uz + fz * uy);
            break;
        case 7: // Viscous stress xx-yy
            // m*_7 = (1-S_7) m_7 + S_7 m^eq_7 + (1-0.5 S_7) F_7
            // m^eq_7 = rho * (ux^2 - uy^2)
            // F_7 = 2 fx ux - 2 fy uy
            m_post_val = (1.0f - S[7]) * m7 + S[7] * rho * (ux2 - uy2) +
                         (1.0f - 0.5f * S[7]) * (2.0f * fx * ux - 2.0f * fy * uy);
            break;
        case 8: // Viscous stress xx-zz
            // m*_8 = (1-S_8) m_8 + S_8 m^eq_8 + (1-0.5 S_8) F_8
            // m^eq_8 = rho * (ux^2 - uz^2)
            // F_8 = 2 fx ux - 2 fz uz
            m_post_val = (1.0f - S[8]) * m8 + S[8] * rho * (ux2 - uz2) +
                         (1.0f - 0.5f * S[8]) * (2.0f * fx * ux - 2.0f * fz * uz);
            break;

        // Equilibrium moments from De Rosis (2019), Table E.1
        // Forcing terms F_i from De Rosis (2019), Appendix E.
        // m*_i = m^eq_i + 0.5 * F_i for these moments based on original code
        case 9: // Energy
            // m^eq_9 = rho (ux^2 + uy^2 + uz^2 + 1)
            // F_9 = 2 fx ux + 2 fy uy + 2 fz uz
            m_post_val = rho * (ux2 + uy2 + uz2 + 1.0f) +
                         0.5f * (2.0f * fx * ux + 2.0f * fy * uy +
                                  2.0f * fz * uz);
            break;
        case 10:
            // m^eq_10 = rho cs^2 ux (3 uy^2 + 3 uz^2 + 2)
            // F_10 = (2/3) fx
            m_post_val = rho * cs2 * ux * (3.0f*uy2 + 3.0f*uz2 + 2.0f) +
                         0.5f * ((2.0f / 3.0f) * fx);
            break;
        case 11:
            // m^eq_11 = rho cs^2 uy (3 ux^2 + 3 uz^2 + 2)
            // F_11 = (2/3) fy
            m_post_val = rho * cs2 * uy * (3.0f*ux2 + 3.0f*uz2 + 2.0f) +
                         0.5f * ((2.0f / 3.0f) * fy);
            break;
        case 12:
            // m^eq_12 = rho cs^2 uz (3 ux^2 + 3 uy^2 + 2)
            // F_12 = (2/3) fz
            m_post_val = rho * cs2 * uz * (3.0f*ux2 + 3.0f*uy2 + 2.0f) +
                         0.5f * ((2.0f / 3.0f) * fz);
            break;
        case 13:
            // m^eq_13 = rho ux (uy^2 - uz^2)
            // F_13 = 0
            m_post_val = rho * ux * (uy2 - uz2) + 0.5f * 0.0f;
            break;
        case 14:
            // m^eq_14 = rho uy (ux^2 - uz^2)
            // F_14 = 0
            m_post_val = rho * uy * (ux2 - uz2) + 0.5f * 0.0f;
            break;
        case 15:
            // m^eq_15 = rho uz (ux^2 - uy^2)
            // F_15 = 0
            m_post_val = rho * uz * (ux2 - uy2) + 0.5f * 0.0f;
            break;
        case 16:
            // m^eq_16 = rho ux uy uz
            // F_16 = 0
            m_post_val = rho * ux * uy * uz + 0.5f * 0.0f;
            break;
        case 17: // Energy related
            // m^eq_17 = rho cs^2 (3(ux^2 uy^2 + ux^2 uz^2 + uy^2 uz^2) + 2(ux^2 + uy^2 + uz^2) + 1)
            // F_17 = (4/3) fx ux + (4/3) fy uy + (4/3) fz uz
            m_post_val = rho * cs2 * (3.0f*(ux2*uy2 + ux2*uz2 + uy2*uz2) +
                                     2.0f*(ux2 + uy2 + uz2) + 1.0f) +
                         0.5f * ((4.0f/3.0f)*fx*ux + (4.0f/3.0f)*fy*uy +
                                  (4.0f/3.0f)*fz*uz);
            break;
        case 18:
            // m^eq_18 = rho cs^4 (9(ux^2 uy^2 + ux^2 uz^2 - uy^2 uz^2) + 6 ux^2 + 1)
            // F_18 = (4/3) fx ux
            m_post_val = rho * cs2 * cs2 * (9.0f*(ux2*uy2 + ux2*uz2 - uy2*uz2) +
                                          6.0f*ux2 + 1.0f) +
                         0.5f * ((4.0f/3.0f)*fx*ux);
            break;
        case 19:
            // m^eq_19 = rho cs^2 (uy^2 - uz^2) (2 ux^2 + 1)
            // F_19 = (2/3) fy uy - (2/3) fz uz
            m_post_val = rho * cs2 * (uy2 - uz2) * (2.0f*ux2 + 1.0f) +
                         0.5f * ((2.0f/3.0f)*fy*uy - (2.0f/3.0f)*fz*uz);
            break;
        case 20:
            // m^eq_20 = rho cs^2 uy uz (3 ux^2 + 1)
            // F_20 = (1/3) fy uz + (1/3) fz uy
            m_post_val = rho * cs2 * uy * uz * (3.0f*ux2 + 1.0f) +
                         0.5f * ((1.0f/3.0f)*fy*uz + (1.0f/3.0f)*fz*uy);
            break;
        case 21:
            // m^eq_21 = rho cs^2 ux uz (3 uy^2 + 1)
            // F_21 = (1/3) fx uz + (1/3) fz ux
            m_post_val = rho * cs2 * ux * uz * (3.0f*uy2 + 1.0f) +
                         0.5f * ((1.0f/3.0f)*fx*uz + (1.0f/3.0f)*fz*ux);
            break;
        case 22:
            // m^eq_22 = rho cs^2 ux uy (3 uz^2 + 1)
            // F_22 = (1/3) fx uy + (1/3) fy ux
            m_post_val = rho * cs2 * ux * uy * (3.0f*uz2 + 1.0f) +
                         0.5f * ((1.0f/3.0f)*fx*uy + (1.0f/3.0f)*fy*ux);
            break;
        case 23:
            // m^eq_23 = rho cs^4 ux (3 uy^2 + 1) (3 uz^2 + 1)
            // F_23 = (1/9) fx
            m_post_val = rho * cs2 * cs2 * ux * (3.0f*uy2 + 1.0f) *
                                             (3.0f*uz2 + 1.0f) +
                         0.5f * ((1.0f/9.0f)*fx);
            break;
        case 24:
            // m^eq_24 = rho cs^4 uy (3 ux^2 + 1) (3 uz^2 + 1)
            // F_24 = (1/9) fy
            m_post_val = rho * cs2 * cs2 * uy * (3.0f*ux2 + 1.0f) *
                                             (3.0f*uz2 + 1.0f) +
                         0.5f * ((1.0f/9.0f)*fy);
            break;
        case 25:
            // m^eq_25 = rho cs^4 uz (3 ux^2 + 1) (3 uy^2 + 1)
            // F_25 = (1/9) fz
            m_post_val = rho * cs2 * cs2 * uz * (3.0f*ux2 + 1.0f) *
                                             (3.0f*uy2 + 1.0f) +
                         0.5f * ((1.0f/9.0f)*fz);
            break;
        case 26: // Energy flux related
            // m^eq_26 = rho cs^6 (3 ux^2 + 1) (3 uy^2 + 1) (3 uz^2 + 1)
            // F_26 = (2/9) fx ux + (2/9) fy uy + (2/9) fz uz
            m_post_val = rho * cs2 * cs2 * cs2 * (3.0f*ux2 + 1.0f) *
                                                (3.0f*uy2 + 1.0f) *
                                                (3.0f*uz2 + 1.0f) +
                         0.5f * ((2.0f/9.0f)*fx*ux + (2.0f/9.0f)*fy*uy +
                                  (2.0f/9.0f)*fz*uz);
            break;
        default:
            // Should not happen with valid moment indices (0-26 for D3Q27)
            m_post_val = -99.0f; // Sentinel value for error
            break;
    }

    return m_post_val;
}

// De Rosis Universal Formulation of Central Moments (2019) Appendix E.
template<>
__device__
void MRT<3>::apply(float* f, float* f_eq, float* u, float* force, int node) {
    float ux = u[get_vec_index(node, 0)];
    float uy = u[get_vec_index(node, 1)];
    float uz = u[get_vec_index(node, 2)];

    float fx = force[get_vec_index(node, 0)];
    float fy = force[get_vec_index(node, 1)];
    float fz = force[get_vec_index(node, 2)];
    
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
    
    for (int k_init = 0; k_init < quadratures; k_init++) {
        f[get_node_index(node, k_init)] = 0.0f;
    }

    for (int i = 0; i < quadratures; i++) {

        // float F_i = compute_forcing_term_component(
        //         i, ux, uy, uz, fx, fy, fz);
        // float m_post_i = compute_m_post_component(
        //         i, rho, ux, uy, uz, m4, m5, m6, m7,
        //         m8, F_i);

        float m_post_i = compute_m_post_combined(i, rho, ux, uy, uz, fx, fy, fz, m4, m5, m6, m7, m8);

        for (int k = 0; k < quadratures; k++) {
            f[get_node_index(node, k)] += M_inv[k * quadratures + i] * m_post_i;
        }
    }

    for (int k_final = 0; k_final < quadratures; ++k_final) {
        int f_idx_final = get_node_index(node, k_final);
        if (fabsf(f[f_idx_final]) > VALUE_THRESHOLD || f[f_idx_final] < -0.01f) {
            int x, y, z_coord;
            get_coords_from_node(node, x, y, z_coord);
            if (node == DEBUG_NODE) {
                printf("[WARNING][MRT<3>::apply] Node %d (x=%d,y=%d,z=%d), Dir %d: f_final[%d] = %f\n",
                       node, x, y, z_coord, k_final, f_idx_final, f[f_idx_final]);
            }
        }
    }
}
