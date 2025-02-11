#ifndef OUTFLOW_H
#define OUTFLOW_H

#include "lbm_constants.cuh"

struct OutflowBoundary {
    __device__ static void apply(float* f, float* f_back, const int* C, const int* OPP, int node) {
        node = get_node_index(node);
        float rho = (f[0 + node] + f[1 + node] + f[3 + node] + f[4 + node] +
                     2.0f * (f[2 + node] + f[6 + node] + f[5 + node]));

        f[8 + node] = f[6 + node] - (1.0f / 6.0f) * rho;
        f[7 + node] = f[5 + node] - (1.0f / 6.0f) * rho;
    }
};

#endif // !OUTFLOW
