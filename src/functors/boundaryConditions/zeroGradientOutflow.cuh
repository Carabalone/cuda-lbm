#ifndef OUTFLOW_H
#define OUTFLOW_H

#include "lbm_constants.cuh"

struct OutflowBoundary {
    __device__ static void apply(float* f, float* f_back, int node) {
        int baseIdx = get_node_index(node, 0);
        
        int x = node % NX;
        int y = node / NX;

        int interior_node = y * NX + (NX - 2);
        int baseIdx_interior = get_node_index(interior_node, 0);
        
        f[baseIdx + 3] = f[baseIdx_interior + 3];
        f[baseIdx + 6] = f[baseIdx_interior + 6];
        f[baseIdx + 7] = f[baseIdx_interior + 7];
    }
};

#endif // !OUTFLOW
