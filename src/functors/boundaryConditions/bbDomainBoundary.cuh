#ifndef BBDOMAIN_BOUNDARY_H
#define BBDOMAIN_BOUNDARY_H

#include "lbm_constants.cuh"


// TODO: I could probably just ignore the x and y part, and just make separate cases for top
// and bottom boundaries, one for left/right boundaries and get rid of the loop and branching
// I am going to redo this when we start dealing with domain boundaries in separate kernels.

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

            bool is_x_boundary = this->x && (x_neigh < 0 || x_neigh >= NX);
            bool is_y_boundary = this->y && (y_neigh < 0 || y_neigh >= NY);
            
            if (is_x_boundary || is_y_boundary) {
                int opp_i = OPP[i];
                
                f[base_idx + opp_i] = saved[i];

                if (saved[i] < 0.0f) {
                    printf("[BOUNCE_BACK] Bounced negative value at node (%d, %d): from dir %d to dir %d, value = %f\n", 
                           x, y, i, opp_i, saved[i]);
                }
            }
        }
    }
};

#endif // ! BBDOMAIN_BOUNDARY_H
