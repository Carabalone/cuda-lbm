#ifndef CYLINDER_BOUNDARY_H
#define CYLINDER_BOUNDARY_H

#include "core/lbm_constants.cuh"

struct CylinderBoundary {
    float cx, cy, r;
    
    __host__ __device__
    CylinderBoundary(float _cx, float _cy, float _r) : cx(_cx), cy(_cy), r(_r) { }
    
    __host__ __device__
    bool is_boundary(int node) const {
        int x = node % NX;
        int y = node / NX;
        float dx = x - cx;
        float dy = y - cy;
        return (dx * dx + dy * dy) <= (r * r);
    }
    
    __host__ __device__
    static inline void apply(float* f, float* f_back, int node) {
        int baseIdx = get_node_index(node, 0);

        float saved[quadratures];
        for (int j = 0; j < quadratures; j++) {
            saved[j] = f[baseIdx + j];
        }

        for (int i = 0; i < quadratures; i++) {
            int opp_i = OPP[i];
            f[baseIdx + opp_i] = saved[i];
        }
    }
};

#endif // CYLINDER_BOUNDARY_H
