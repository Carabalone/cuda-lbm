#ifndef GUO_INLET_3D_H
#define GUO_INLET_3D_H
#include "core/lbm_constants.cuh"
#include "util/utility.cuh"

struct GuoVelocityInlet3D
{
    __device__ static inline void apply(float* f, float* rho, float* u, int node, float u_inlet) {
        // This boundary condition is for an inlet at x=0.
        // The unknown populations are those with c_ix > 0.
        
        // --- Step 1: Calculate boundary density from known populations ---
        // This is the crucial step for mass conservation.
        float sum_known_f = 0.0f;
        #pragma unroll
        for (int i = 0; i < quadratures; ++i) {
            // Sum all populations that are known after streaming (i.e., coming from the fluid domain)
            if (C[i * dimensions + 0] < 0) { // c_ix < 0
                sum_known_f += 2.0f * f[get_node_index(node, i)];
            } else if (C[i * dimensions + 0] == 0) { // c_ix = 0
                sum_known_f += f[get_node_index(node, i)];
            }
        }

        // Prescribe the velocity at the boundary
        const float ux_b = u_inlet;
        const float uy_b = 0.0f;
        const float uz_b = 0.0f;

        // Calculate the boundary density using the consistent formula
        const float rho_b = sum_known_f / (1.0f - ux_b);

        // --- Step 2: Compute equilibrium and non-equilibrium parts ---
        // Equilibrium at the boundary node based on prescribed u and calculated rho
        float feq_b[quadratures];
        de_rosis_eq(feq_b, ux_b, uy_b, uz_b, rho_b);

        // Get the adjacent fluid node to extrapolate the non-equilibrium part from
        int x, y, z;
        get_coords_from_node(node, x, y, z);
        int fluid_node = get_node_from_coords(x + 1, y, z);


        float rho_f = 0.0f;
        float ux_f = 0.0f;
        float uy_f = 0.0f;
        float uz_f = 0.0f;

        for (int i=0; i < quadratures; i++) {
            float f_i = f[get_node_index(node, i)];
            rho_f += f_i;
            ux_f  += f_i * C[3*i];
            uy_f  += f_i * C[3*i + 1];
            uz_f  += f_i * C[3*i + 2];
        }

        ux_f  /= rho_f;
        uy_f  /= rho_f;
        uz_f  /= rho_f;
        
        float feq_f[quadratures];
        de_rosis_eq(feq_f, ux_f, uy_f, uz_f, rho_f);

        // --- Step 3: Reconstruct unknown populations ---
        #pragma unroll
        for (int i = 0; i < quadratures; ++i) {
            if (C[i * dimensions + 0] > 0) { // c_ix > 0 are the unknown populations
                // Extrapolate the non-equilibrium part from the fluid neighbor
                float f_neq_fluid = f[get_node_index(fluid_node, i)] - feq_f[i];
                f[get_node_index(node, i)] = feq_b[i] + f_neq_fluid;
            }
        }
    }


 private:
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
#endif
