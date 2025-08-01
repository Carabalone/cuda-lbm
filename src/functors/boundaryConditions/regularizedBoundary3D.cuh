#ifndef REGULARIZED_BOUNDARY_H
#define REGULARIZED_BOUNDARY_H

#include "core/lbm_constants.cuh"
#include "util/utility.cuh"

// Important to note that this is domain only. Given that, inlets and no-slip boundaries just differ from the expected velocity at the
// boudary (e.g. inlet left for a velocity with dir (1,0,0) is RegularizedBounceBack::apply(f, node, u_x, 0, 0) on a node at x=0). We still hold problems with corners, but that's for another time.
struct RegularizedBoundary {

    __device__ static inline void apply(float* f, int node, float ux, float uy, float uz,
                                        int ramp_up_time, int ts) {
        int x, y, z;
        get_coords_from_node(node, x, y, z);

        int n_dim  = -1;   // 0-x, 1-y, 2-z
        int n_sign =  0;   // +1 = min face, -1 = max face
        if (x == 0      ) { n_dim = 0; n_sign = +1; }
        if (x == NX - 1 ) { n_dim = 0; n_sign = -1; }
        if (y == 0      ) { n_dim = 1; n_sign = +1; }
        if (y == NY - 1 ) { n_dim = 1; n_sign = -1; }
        if (z == 0      ) { n_dim = 2; n_sign = +1; }
        if (z == NZ - 1 ) { n_dim = 2; n_sign = -1; }
        if (n_dim == -1){
            printf("n_dim==-1\n");
            return; // somehow at interior  
        }

        float rho_incoming = 0.f;
        float rho_parallel = 0.f;

        for (int i=0; i < quadratures; i++) {
            // normal has 1 one and rest is zeroes
            // n dot C = c[normal_dim] * n_sign
            if (is_known(i, n_dim, n_sign)) {
                float c_dot_n = C[3*i + n_dim] * n_sign;
                // if (y == 12 && z == 10)
                    // printf("known: %d (%d, %d, %d) | parallel: %s\n", i, C[3*i], C[3*i+1], C[3*i+2], (fabsf(c_dot_n) < 1e-3f ? "true" : "false"));
                float f_i = f[get_node_index(node, i)];

                if (fabsf(c_dot_n) < 1e-3f) rho_parallel  += f_i;
                else if  (c_dot_n  < 0.f)   rho_incoming  += f_i;
                else printf("[REGULARIZED_INLET]: unknown direction\n");
                

            }
        }
        float rho = rho_parallel + 2.0f*rho_incoming;

        float u_n = (n_dim == 0) ? ux :
                    (n_dim == 1) ? uy :
                    uz;
        
        u_n = utl::clamp_ramp_up_linear(u_n, ramp_up_time, ts);

        if      (n_dim == 0) ux = u_n;
        else if (n_dim == 1) uy = u_n;
        else                 uz = u_n;

        // float rho = (2.0f*rho_incoming + rho_parallel + 1.0f) / (1.0f + u_n);
        rho /= (1 - n_sign * u_n);
        // float rho = rho_parallel + rho_incoming;

        if (utl::in_range(y, 50, 56) && utl::in_range(z, 40, 48)) {
            // printf("calc_rho: %.4f\n", rho);
        }

        // if (fabsf(rho) > 1.08f || fabsf(rho) < 0.92f)
        // // // if (x == 132 && y == 0 && z == 32)
        //     printf("Rho @ regularized boundary (%d, %d, %d): %.4f\n", x, y, z, rho);

        
        // equilibrium using imposed velocity
        float f_eq[quadratures];
        float cs2 = 1.0f / 3.0f;

        // TODO: IDK about this man, just trying to not let mass into the system.
        // rho = 1.0f;
        de_rosis_eq(f_eq, ux, uy, uz, rho);

        //  bounce-back of non-equilibrium parts
        for (int q = 0; q < quadratures; q++) {
            if (!is_known(q, n_dim, n_sign)) {
                int opp = OPP[q];
                // f_neq = f - f_eq
                f[get_node_index(node, q)] = f_eq[q] + (f[get_node_index(node, opp)] - f_eq[opp]);
            }
        }
     
        // // Π(1) = ∑ᵢ fᵢ - cs² ρ I - ρ uu
        float Pi_xx = 0.0f, Pi_yy = 0.0f, Pi_zz = 0.0f, Pi_xy = 0.0f, Pi_xz = 0.0f, Pi_yz = 0.0f;
        
        for (int q = 0; q < quadratures; q++) {
            float fi = f[get_node_index(node, q)];
            float cx = C[q * 3 + 0];
            float cy = C[q * 3 + 1];
            float cz = C[q * 3 + 2];

            Pi_xx += cx*cx * fi;
            Pi_yy += cy*cy * fi;
            Pi_zz += cz*cz * fi;
            Pi_xy += cx*cy * fi;
            Pi_xz += cx*cz * fi;
            Pi_yz += cy*cz * fi;    
        }

        Pi_xx -= cs2 * rho + rho * ux * ux;
        Pi_yy -= cs2 * rho + rho * uy * uy;
        Pi_zz -= cs2 * rho + rho * uz * uz;
        Pi_xy -= rho * ux * uy;
        Pi_xz -= rho * ux * uz;
        Pi_yz -= rho * uy * uz;
        
        // float test_rho = 0.0f;
        // float test_u[3] = {0.0f};
        for (int q = 0; q < quadratures; q++) {
            float cx = C[3*q], cy = C[3*q+1], cz = C[3*q+2];
            const float Q_xx = cx*cx - cs2;
            const float Q_yy = cy*cy - cs2;
            const float Q_zz = cz*cz - cs2;
            const float Q_xy = cx*cy;
            const float Q_xz = cx*cz;
            const float Q_yz = cy*cz;
            
            float f_neq = WEIGHTS[q] / (2.0f*cs2*cs2) *
               (Q_xx*Pi_xx + Q_yy*Pi_yy + Q_zz*Pi_zz +
                 2.0f*(Q_xy*Pi_xy + Q_xz*Pi_xz + Q_yz*Pi_yz));
            
            f[get_node_index(node, q)] = f_eq[q] + f_neq;
            float f_i = f_eq[q] + f_neq;
            // test_u[0] += cx * f_i;
            // test_u[1] += cy * f_i;
            // test_u[2] += cz * f_i;
            // test_rho += f_i;
        }

        // if (x == 0 && utl::in_range(y, 32, 96) && utl::in_range(z, 32, 96) && ts > 400){
        //     printf("test_rho after regularization @ (%d, %d, %d): %.4f|u (%.4f, %.4f, %.4f)\n", x, y, z, test_rho,
        //             test_u[0], test_u[1], test_u[2]);
        // }        
    }

private:
    __device__
    static inline bool is_known(int i, int ndim, int nsign) { return C[3*i + ndim] * nsign <= 0.0f; }

    __device__
    static void de_rosis_eq(float* f_eq, float ux, float uy, float uz, float rho) {
        float ux2 = ux * ux;
        float uy2 = uy * uy;
        float uz2 = uz * uz;
        float u_dot_u = ux2 + uy2 + uz2;
        float cs2 = 1.0f/3.0f;
        float cs4 = cs2 * cs2;
        float two_cs4 = 2.0f * cs4;
        float cs2_cs4 = cs2 * cs4;
        float inv_cs2_cs4 = 1.0f / cs2_cs4;
        float two_cs2_cs4 = 2.0f * cs2 * cs4;
        float half_rho = 0.5f * rho;

        // De Rosis (2017)
        f_eq[0]  = half_rho*WEIGHTS[0]*(2.0f*cs2 - ux2 - uy2 - uz2)/cs2;
        f_eq[1]  = inv_cs2_cs4*half_rho*WEIGHTS[1]*(1.0f*cs2*ux2 + two_cs2_cs4 + two_cs4*ux - cs4*(ux2 + uy2 + uz2));
        f_eq[2]  = inv_cs2_cs4*half_rho*WEIGHTS[2]*(1.0f*cs2*ux2 + two_cs2_cs4 - two_cs4*ux - cs4*(ux2 + uy2 + uz2));
        f_eq[3]  = inv_cs2_cs4*half_rho*WEIGHTS[3]*(1.0f*cs2*uy2 + two_cs2_cs4 + two_cs4*uy - cs4*(ux2 + uy2 + uz2));
        f_eq[4]  = inv_cs2_cs4*half_rho*WEIGHTS[4]*(1.0f*cs2*uy2 + two_cs2_cs4 - two_cs4*uy - cs4*(ux2 + uy2 + uz2));
        f_eq[5]  = inv_cs2_cs4*half_rho*WEIGHTS[5]*(1.0f*cs2*uz2 + two_cs2_cs4 + two_cs4*uz - cs4*(ux2 + uy2 + uz2));
        f_eq[6]  = inv_cs2_cs4*half_rho*WEIGHTS[6]*(1.0f*cs2*uz2 + two_cs2_cs4 - two_cs4*uz - cs4*(ux2 + uy2 + uz2));
        f_eq[7]  = inv_cs2_cs4*half_rho*WEIGHTS[7]*(1.0f*cs2 *(ux + uy)*((ux + uy) + two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[8]  = inv_cs2_cs4*half_rho*WEIGHTS[8]*(1.0f*cs2 *(ux - uy)*((ux - uy) - two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[9]  = inv_cs2_cs4*half_rho*WEIGHTS[9]*(1.0f*cs2 *(ux - uy)*((ux - uy) + two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[10] = inv_cs2_cs4*half_rho*WEIGHTS[10]*(1.0f*cs2*(ux + uy)*((ux + uy) - two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[11] = inv_cs2_cs4*half_rho*WEIGHTS[11]*(1.0f*cs2*(ux + uz)*((ux + uz) + two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[12] = inv_cs2_cs4*half_rho*WEIGHTS[12]*(1.0f*cs2*(ux - uz)*((ux - uz) - two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[13] = inv_cs2_cs4*half_rho*WEIGHTS[13]*(1.0f*cs2*(ux - uz)*((ux - uz) + two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[14] = inv_cs2_cs4*half_rho*WEIGHTS[14]*(1.0f*cs2*(ux + uz)*((ux + uz) - two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[15] = inv_cs2_cs4*half_rho*WEIGHTS[15]*(1.0f*cs2*(uy + uz)*((uy + uz) + two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[16] = inv_cs2_cs4*half_rho*WEIGHTS[16]*(1.0f*cs2*(uy - uz)*((uy - uz) - two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[17] = inv_cs2_cs4*half_rho*WEIGHTS[17]*(1.0f*cs2*(uy - uz)*((uy - uz) + two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[18] = inv_cs2_cs4*half_rho*WEIGHTS[18]*(1.0f*cs2*(uy + uz)*((uy + uz) - two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[19] = inv_cs2_cs4*half_rho*WEIGHTS[19]*(1.0f*cs2*(ux + uy + uz)*((ux + uy + uz) + two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
        f_eq[20] = inv_cs2_cs4*half_rho*WEIGHTS[20]*(1.0f*cs2*(-ux + uy + uz)*((-ux + uy + uz) + two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
        f_eq[21] = inv_cs2_cs4*half_rho*WEIGHTS[21]*(1.0f*cs2*(ux - uy + uz)*((ux - uy + uz) + two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
        f_eq[22] = inv_cs2_cs4*half_rho*WEIGHTS[22]*(1.0f*cs2*(ux + uy - uz)*((ux + uy - uz) - two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
        f_eq[23] = inv_cs2_cs4*half_rho*WEIGHTS[23]*(1.0f*cs2*(ux + uy - uz)*((ux + uy - uz) + two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
        f_eq[24] = inv_cs2_cs4*half_rho*WEIGHTS[24]*(1.0f*cs2*(ux - uy + uz)*((ux - uy + uz) - two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
        f_eq[25] = inv_cs2_cs4*half_rho*WEIGHTS[25]*(1.0f*cs2*(-ux + uy + uz)*((-ux + uy + uz) - two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
        f_eq[26] = inv_cs2_cs4*half_rho*WEIGHTS[26]*(1.0f*cs2*(ux + uy + uz)*((ux + uy + uz) - two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
    }

};

struct VelocityInlet {
    __device__ static void apply(float* f, int node, float u_max, int ramp_time, int ts) {
        int x, y, z;
        get_coords_from_node(node, x, y, z);
        
        float y_center = (NY - 1) * 0.5f;
        float z_center = (NZ - 1) * 0.5f;
        float r_y = (y - y_center) / y_center;
        float r_z = (z - z_center) / z_center;
        float r2 = r_y*r_y + r_z*r_z;
        
        float u_profile = u_max * (1.0f - r2);
        u_profile = utl::clamp_ramp_up_tanh(u_profile, ramp_time, ts);
        
        RegularizedBoundary::apply(f, node, u_profile, 0.0f, 0.0f, ramp_time, ts);
    }
};

#endif // !REGULARIZED_BOUNDARY_H
