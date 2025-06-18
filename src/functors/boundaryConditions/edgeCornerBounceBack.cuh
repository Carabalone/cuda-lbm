
#ifndef EDGE_CORNER_BOUNCE_BACK_BOUNDARY_H
#define EDGE_CORNER_BOUNCE_BACK_BOUNDARY_H

#include "core/lbm_constants.cuh"

struct EdgeCornerBounceBack {
    __host__ __device__
    EdgeCornerBounceBack() {}
    
    __device__
    static inline void apply(float* f, int node) {
        int x, y, z;
        get_coords_from_node(node, x, y, z);
        
        int normals[3] = {0};
        if      (x == 0)    normals[0] =  1;
        else if (x == NX-1) normals[0] = -1;
        
        if      (y == 0)    normals[1] =  1;
        else if (y == NY-1) normals[1] = -1;
        
        if      (z == 0)    normals[2] =  1;
        else if (z == NZ-1) normals[2] = -1;
        
        int num_normals = 0;
        #pragma unroll
        for (int d = 0; d < dimensions; d++) {
            if (normals[d] != 0) num_normals++;
        }
        
        if (num_normals < 2) return;

        for (int i = 0; i < quadratures; i++) {
            const int opp = OPP[i];
            const int cx = C[i*3];
            const int cy = C[i*3+1];
            const int cz = C[i*3+2];
            const int ox = C[opp*3];
            const int oy = C[opp*3+1];
            const int oz = C[opp*3+2];
            
            const bool i_out = (NX*cx > 0) || (NY*cy > 0) || (NZ*cz > 0);
            const bool o_out = (NX*ox > 0) || (NY*oy > 0) || (NZ*oz > 0);
            
            if (i_out && !o_out) {
                f[get_node_index(node, opp)] = f[get_node_index(node, i)];
            }
        }
    }
};
#endif // ! EDGE_CORNER_BOUNCE_BACK_BOUNDARY_H
