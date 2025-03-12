#ifndef REGULARIZED_INLET_H
#define REGULARIZED_INLET_H

#include "core/lbm_constants.cuh"

// Latt, B. Chopard, O. Malaspinas, M. Deville, and A. Michler, Phys. Rev. E 77, 056703 (2008)
// regularized boundaries used by De Rosis to achieve Re=6000 in the Lid-driven Cavity using BGK in
// Nonorthogonal central-moments-based lattice Boltzmann scheme in three dimensions. 

// TODO: we can make a generalized version of this. This works in D3Q27 which is a major advantage
// TODO: compared to Zou/He. I will do it only when I need it though
//TODO: Reduce reused code from the two conditions here and in regularized BounceBack
struct RegularizedInlet {
    __device__
    static void apply_top(float* f, float* f_back, float u_lid, int node) {
        int idx = get_node_index(node);
        const float ux = u_lid;
        const float uy = 0.0f;
        
        float rho = (f[idx + 0] + f[idx + 1] + f[idx + 3] + 
                    2.0f * (f[idx + 2] + f[idx + 5] + f[idx + 6])) / (1.0f + uy);
        

        // equilibrium using imposed velocity
        float f_eq[quadratures];
        float cs2 = 1.0f / 3.0f;
        for (int q = 0; q < quadratures; q++) {
            float c_dot_u = C[2*q] * ux + C[2*q+1] * uy;
            float u_dot_u = ux*ux + uy*uy;
            f_eq[q] = WEIGHTS[q] * rho * (1.0f + c_dot_u/cs2 + 
                        c_dot_u*c_dot_u/(2.0f*cs2*cs2) - u_dot_u/(2.0f*cs2));
        }
        
        //  bounce-back of non-equilibrium parts
        for (int q = 0; q < quadratures; q++) {
            if (C[2*q+1] < 0) {
                int opp = OPP[q];
                f[idx + q] = f_eq[q] + (f[idx + opp] - f_eq[opp]);
            }
        }

        // Π(1) = ∑ᵢ fᵢ - cs² ρ I - ρ uu
        float Pi_xx = 0.0f, Pi_yy = 0.0f, Pi_xy = 0.0f;
        
        for (int q = 0; q < quadratures; q++) {
            float cx = C[2*q], cy = C[2*q+1];
            Pi_xx += cx * cx * f[idx + q];
            Pi_yy += cy * cy * f[idx + q];
            Pi_xy += cx * cy * f[idx + q];
        }

        Pi_xx -= cs2 * rho + rho * ux * ux;
        Pi_yy -= cs2 * rho + rho * uy * uy;
        Pi_xy -= rho * ux * uy;

        // regularize all distributions
        for (int q = 0; q < quadratures; q++) {
            float cx = C[2*q], cy = C[2*q+1];
            float Q_xx = cx*cx - cs2;
            float Q_yy = cy*cy - cs2;
            float Q_xy = cx*cy;
            
            float f_neq = (WEIGHTS[q]/(2.0f*cs2*cs2)) * 
                        (Q_xx*Pi_xx + Q_yy*Pi_yy + 2.0f*Q_xy*Pi_xy);
            
            f[idx + q] = f_eq[q] + f_neq;
        }
                    
    }
};

#endif // ! REGULARIZED_INLET_H