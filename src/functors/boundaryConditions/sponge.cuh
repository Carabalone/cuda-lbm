#ifndef SPONGE_FUNCTOR_H
#define SPONGE_FUNCTOR_H

#include "core/lbm_constants.cuh"
#include "util/utility.cuh"

struct Sponge {
    __device__ static inline void apply(float* f, int node, int Ns, float sigma_max) {
        int x,y,z; get_coords_from_node(node,x,y,z);
        int start = NX - Ns;
        int d = x - start;
        if (d<0 || d>=Ns) return;

        float xi = float(d)/Ns;
        float sigma = sigma_max * xi*xi;

        float local_rho = 0.f;
        float ux = 0.f, uy = 0.f, uz = 0.f;
        #pragma unroll
        for (int q=0; q<quadratures; ++q) {
            float fi = f[get_node_index(node,q)];
            local_rho += fi;
            ux += fi * C[3*q + 0];
            uy += fi * C[3*q + 1];
            uz += fi * C[3*q + 2];
        }
        ux /= local_rho; 
        uy /= local_rho; 
        uz /= local_rho;

        float feq_star[quadratures];
        compute_equilibrium(feq_star, ux, uy, uz, local_rho);

        #pragma unroll
        for (int q=0; q<quadratures; ++q) {
            int idx = get_node_index(node,q);
            f[idx] += -sigma * (f[idx] - feq_star[q]);
        }
    }

    // 2nd order is cheaper and enough for sponges.
    __device__ static inline void compute_equilibrium(float* feq, float ux, float uy, float uz, float rho) {
        const float cs2 = 1.f/3.f;
        const float cs4 = cs2*cs2;
        float u2 = ux*ux + uy*uy + uz*uz;
        #pragma unroll
        for (int q=0; q<quadratures; ++q) {
            float cx = C[3*q+0], cy = C[3*q+1], cz = C[3*q+2];
            float c_dot_u = cx*ux + cy*uy + cz*uz;
            feq[q] = WEIGHTS[q]*rho*
              (1.f + c_dot_u/cs2
                  + 0.5f*(c_dot_u*c_dot_u)/(cs4)
                  - 0.5f*u2/cs2);
        }
    }
};

#endif  // SPONGE_FUNCTOR_H
