#ifndef BBDOMAIN_BOUNDARY_H
#define BBDOMAIN_BOUNDARY_H

#include "lbm_constants.cuh"

struct BBDomainBoundary {
    bool x, y;

    __host__ __device__
    BBDomainBoundary(bool x=true, bool y=true) : x(x), y(y) {}

    __device__
    inline void apply(float* f, float* f_back, int node) {
        int base_idx = get_node_index(node);
        const int x = node % NX;
        const int y = node / NX;

        float saved[quadratures];
        for (int j = 0; j < quadratures; j++) {
            saved[j] = f[base_idx + j];
        }
        
        for (int i = 1; i < quadratures; i++) {
            int x_neigh = x + C[2 * i];
            int y_neigh = y + C[2 * i + 1];
            if ((x && (x_neigh < 0 || x_neigh >= NX)) ||
                (y && (y_neigh < 0 || y_neigh >= NY))) {
        
                int opp_i = OPP[i];
                
                // if (node == 0) {
                //     printf("  Applying boundary condition for direction %d:\n", i);
                //     printf("    f[%d] = %f, f_back[%d] = %f\n",
                //            base_idx + i, f[base_idx + i], base_idx + opp_i, f_back[base_idx + opp_i]);
                // }

                // f_back[base_idx + opp_i] = f_back[base_idx + i];
                f[base_idx + opp_i] = saved[i];
            } 
        }
    }
};

#endif // ! BBDOMAIN_BOUNDARY_H
