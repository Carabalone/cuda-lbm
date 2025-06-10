#ifndef OUTFLOW_H
#define OUTFLOW_H

#include "core/lbm_constants.cuh"

template <int dim>
struct ZG_OutflowBoundary {

    __device__ static inline void apply(float* f, int node) {
        int x, y, z;
        get_coords_from_node(node, x, y, z);

        int normal[dim] = {0};
        int interior_node_coord[dim] = {x, y, z};

        if (x == 0) {
            normal[0] = 1;
            interior_node_coord[0] = 1;
        }
        else if (x == NX - 1) {
            normal[0] = -1;
            interior_node_coord[0] = NX - 2;
        }
        else if (y == 0) {
            normal[1] = 1;
            interior_node_coord[1] = 1;
        }
        else if (y == NY - 1) {
            normal[1] = -1;
            interior_node_coord[1] = NY - 2;
        }
        else if (z == 0) {
            normal[2] = 1;
            interior_node_coord[2] = 1;
        }
        else if (z == NZ - 1) {
            normal[2] = -1;
            interior_node_coord[2] = NZ - 2;
        }
        else {
            return;
        }

        int interior_node = get_node_from_coords(interior_node_coord[0], interior_node_coord[1], interior_node_coord[2]);

        for (int i = 0; i < quadratures; ++i) {
            int c_dot_n = 0;
            #pragma unroll
            for (int d = 0; d < dim; ++d) {
                c_dot_n += C[i * dim + d] * normal[d];
            }

            bool is_unknown = (c_dot_n > 0);
            
            float original = f[get_node_index(node, i)];
            float extended = f[get_node_index(interior_node, i)];
            f[get_node_index(node, i)] = is_unknown ? extended : original;
        }
    }
};

#endif // !OUTFLOW
