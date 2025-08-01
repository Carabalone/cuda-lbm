#ifndef FREE_SLIP_BOUNCE_BACK_H
#define FREE_SLIP_BOUNCE_BACK_H

#include "core/lbm_constants.cuh"

template <int dim>
struct FreeSlipBounceBack {
    __device__ static inline void apply(float* f, int node) {
        int x, y, z;
        get_coords_from_node(node, x, y, z);

        int n_dim = -1;
        if (x == 0 || x == NX-1) n_dim = 0;
        else if (y == 0 || y == NY-1) n_dim = 1;
        else if (dim == 3 && (z == 0 || z == NZ-1)) n_dim = 2;
        else return;

        const int* map = nullptr;
        if (n_dim == 0) map = free_slip_map_x;
        else if (n_dim == 1) map = free_slip_map_y;
        else if (dim == 3 && n_dim == 2) map = free_slip_map_z;

        for (int i = 1; i < quadratures; ++i) {
            int c_normal = C[dim * i + n_dim];
            bool is_out =
                (n_dim == 0 && ((x == 0 && c_normal < 0) || (x == NX-1 && c_normal > 0))) ||
                (n_dim == 1 && ((y == 0 && c_normal < 0) || (y == NY-1 && c_normal > 0))) ||
                (dim == 3 && n_dim == 2 && ((z == 0 && c_normal < 0) || (z == NZ-1 && c_normal > 0)));
            if (is_out) {
                float f_val = f[get_node_index(node, i)];
                f[get_node_index(node, map[i])] = f_val;
            }
        }
    }
};

#endif // FREE_SLIP_BOUNCE_BACK_H
