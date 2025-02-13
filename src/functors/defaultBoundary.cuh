#ifndef DEFAULT_BOUNDARY_H
#define DEFAULT_BOUNDARY_H

#include "lbm_constants.cuh"

struct DefaultBoundary {

    __host__ __device__
    static inline void apply(float* f, float* f_back, int* C, int* OPP, int node) {
        int baseIdx = get_node_index(node, 0);
        const int x = node % NX;
        const int y = node / NX;

        float saved[quadratures];
        for (int j = 0; j < quadratures; j++) {
            saved[j] = f[baseIdx + j];
        }
        
        for (int i = 1; i < quadratures; i++) {
            int x_neigh = x + C[2 * i];
            int y_neigh = y + C[2 * i + 1];
            if (x_neigh < 0 || x_neigh >= NX ||
                y_neigh < 0 || y_neigh >= NY) {
        
                int opp_i = OPP[i];
                
                // if (node == 0) {
                //     printf("  Applying boundary condition for direction %d:\n", i);
                //     printf("    f[%d] = %f, f_back[%d] = %f\n",
                //            baseIdx + i, f[baseIdx + i], baseIdx + opp_i, f_back[baseIdx + opp_i]);
                // }

                // f_back[baseIdx + opp_i] = f_back[baseIdx + i];
                f[baseIdx + opp_i] = saved[i];
            } 
        }
    }
};

#endif // ! DEFAULT_BOUNDARY_H
