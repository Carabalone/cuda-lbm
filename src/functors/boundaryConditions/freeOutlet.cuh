#ifndef REGULARIZED_OUTLET_H
#define REGULARIZED_OUTLET_H

#include "core/lbm_constants.cuh"
#include "util/utility.cuh"

struct RegularizedOutlet {

    __device__ static inline void apply(float* f, float* rho, int node, float ux, float uy, float uz) {
        int x, y, z;
        get_coords_from_node(node, x, y, z);

        int neigh_node = get_node_from_coords(x - 1, y, z);

        rho_neigh = rho[neigh_node];
        ux_n = u[get_node_index(neigh_node, 0)];
        uy_n = u[get_node_index(neigh_node, 1)];
        uz_n = u[get_node_index(neigh_node, 2)];

        float feq_neigh[quadratures];
        de_rosis_eq(feq_neigh, ux_n, uy_n, uz_n, rho_neigh);

        for(int i=0; i < Q; i++){
            if(!is_known(i, 0, -1)){
                int opp = OPP[i];
                float f_neigh_opp = f[get_node_index(neigh_node, opp)];

                f[get_node_index(node,i)] = feq_neigh[i] + (f_neigh_opp - feq_neigh[opp]);
            }
        }

        const float cs2 = 1.0f/3.0f;
        float Pi_xx=0, Pi_yy=0, Pi_zz=0;
        float Pi_xy=0, Pi_xz=0, Pi_yz=0;

        for(int i=0; i < quadratures; ++i){
            float f_i = f[get_node_index(node,i)];
            float cx = C[3*i+0], 
                  cy = C[3*i+1], 
                  cz = C[3*i+2];

            Pi_xx += (cx*cx - cs2)*f_i;
            Pi_yy += (cy*cy - cs2)*f_i;
            Pi_zz += (cz*cz - cs2)*f_i;
            Pi_xy +=  cx*cy * f_i;
            Pi_xz +=  cx*cz * f_i;
            Pi_yz +=  cy*cz * f_i;
        }

        float feq_b[quadratures];
        de_rosis_eq(feq_b, ux_n, uy_n, uz_n, rho_neigh);

        #pragma unroll
        for(int i=0;i<quadratures;++i){
            float cx=C[3*i], cy=C[3*i+1], cz=C[3*i+2];
            float qxx=cx*cx-cs2, qyy=cy*cy-cs2, qzz=cz*cz-cs2;
            float qxy=cx*cy, qxz=cx*cz, qyz=cy*cz;
            float fneq = WEIGHTS[i]/(2.f*cs2*cs2)*(
                qxx*Pi_xx + qyy*Pi_yy + qzz*Pi_zz
              + 2.f*(qxy*Pi_xy + qxz*Pi_xz + qyz*Pi_yz)
            );
            f[get_node_index(node,i)] = feq_b[i] + fneq;
        }

        rho[node] = rho_b;
        u[get_vec_index(node, 0)] = ux_b;
        u[get_vec_index(node, 1)] = uy_b;
        u[get_vec_index(node, 2)] = uz_b;
        

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

#endif // !REGULARIZED_OUTLET_H

