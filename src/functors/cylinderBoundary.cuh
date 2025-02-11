#ifndef CYLINDER_BOUNDARY_H
#define CYLINDER_BOUNDARY_H

#include "defaultBoundary.cuh"
#include "../lbm_constants.cuh"

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
    static inline void apply(float* f, float* f_back, int* C, int* OPP, int node) {
        DefaultBoundary::apply(f, f_back, C, OPP, node);
    }
};

#endif // CYLINDER_BOUNDARY_H
