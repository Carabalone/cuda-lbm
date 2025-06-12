#ifndef REGULARIZED_BOUNDARY_H
#define REGULARIZED_BOUNDARY_H

#include "core/lbm_constants.cuh"

// Important to note that this is domain only. Given that, inlets and no-slip boundaries just differ from the expected velocity at the
// boudary (e.g. inlet left for a velocity with dir (1,0,0) is RegularizedBounceBack::apply(f, node, u_x, 0, 0) on a node at x=0). We still hold problems with corners, but that's for another time.
struct RegularizedBoundary {

    __device__ static inline void apply(float* f, int node, float ux, float uy, float uz) {
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
        if (n_dim == -1) return; // somehow at interior

        float rho_incoming = 0.f;
        float rho_parallel  = 0.f;

        for (int i=0; i < quadratures; i++) {
            // normal has 1 one and rest is zeroes
            // n dot C = c[normal_dim] * n_sign
            if (is_known(i, n_dim, n_sign)) {
                float c_dot_n = C[3*i + n_dim] * n_sign;
                float f_i = f[get_node_index(node, i)];

                if (fabsf(c_dot_n) < 1e-6f) rho_parallel += f_i;
                else if  (c_dot_n  < 0.f)   rho_incoming += f_i;
                

            }
        }
        float rho = rho_parallel + 2.0f*rho_incoming;

        float u_n = (n_dim == 0) ? ux :
                    (n_dim == 1) ? uy :
                    uz;
        rho /= (1 - n_sign * u_n);

        // if (fabsf(rho) > 1.08f || fabsf(rho) < 0.92f)
        if (x == 132 && y == 0 && z == 32)
            printf("Rho @ regularized boundary (%d, %d, %d): %.4f\n", x, y, z, rho);

        
        // equilibrium using imposed velocity
        float f_eq[quadratures];
        float cs2 = 1.0f / 3.0f;

        for (int q=0; q < quadratures; q++) {
            
            float cx = C[q * 3 + 0];
            float cy = C[q * 3 + 1];
            float cz = C[q * 3 + 2];
            float c_dot_u = cx*ux + cy*uy + cz*uz;
            float u_dot_u = ux*ux + uy*uy + uz*uz;
            f_eq[q] = WEIGHTS[q] * rho *
                     (1.f + c_dot_u/cs2 +
                      0.5f*(c_dot_u*c_dot_u)/(cs2*cs2) -
                      0.5f*u_dot_u/cs2);
        }

        
        //  bounce-back of non-equilibrium parts
        for (int q = 0; q < quadratures; q++) {
            if (C[3*q + n_dim] * n_sign > 0) {
                int opp = OPP[q];
                f[get_node_index(node, q)] = f_eq[q] + (f[get_node_index(node, opp)] - f_eq[opp]);
            }
        }
     
        // Π(1) = ∑ᵢ fᵢ - cs² ρ I - ρ uu
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

        
        float test_rho = 0.0f;
        float test_u[3] = {0.0f};
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
            test_u[0] += cx * f_i;
            test_u[1] += cy * f_i;
            test_u[2] += cz * f_i;
            test_rho += f_i;
        }

        if (x == 132 && y == 0 && z == 32){
            printf("test_rho after regularization @ (%d, %d, %d): %.4f|u (%.4f, %.4f, %.4f)\n", x, y, z, test_rho,
                    test_u[0], test_u[1], test_u[2]);
        }        
    }

private:
    __device__
    static inline bool is_known(int i, int ndim, int nsign) { return C[3*i + ndim] * nsign <= 0.0f; }

};

#endif // !REGULARIZED_BOUNDARY_H
