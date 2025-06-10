#ifndef CORNER_H
#define CORNER_H

#include "core/lbm_constants.cuh"

template <int dim>
struct ExtrapolatedCornerEdgeBoundary {

    __device__ static inline void apply(float* f, int node) {
        int x, y, z;
        get_coords_from_node(node, x, y, z);

        int interior_x = (x == 0) ? 1 : (x == NX - 1) ? NX - 2 : x;
        int interior_y = (y == 0) ? 1 : (y == NY - 1) ? NY - 2 : y;
        int interior_z = (z == 0) ? 1 : (z == NZ - 1) ? NZ - 2 : z;
        int interior_node = get_node_from_coords(interior_x, interior_y, interior_z);

        float rho = 0.0f;
        for (int i = 0; i < quadratures; ++i) {
            rho += f[get_node_index(interior_node, i)];
        }

        const float wall_ux = 0.0f, wall_uy = 0.0f, wall_uz = 0.0f;

        for (int i = 0; i < quadratures; ++i) {
            float c_dot_u = C[i * dim + 0] * wall_ux + C[i * dim + 1] * wall_uy + C[i * dim + 2] * wall_uz;
            float u_dot_u = wall_ux * wall_ux + wall_uy * wall_uy + wall_uz * wall_uz;
            f[get_node_index(node, i)] = WEIGHTS[i] * rho * (1.0f + c_dot_u / cs2 + (c_dot_u * c_dot_u) / (2.0f * cs2 * cs2) - u_dot_u / (2.0f * cs2));
        }
    }
};

struct CornerBoundary {
    __device__ static void apply_top_left(float* f, float* f_back, int node) {
        node = get_node_index(node);
        
        const float ux = 0.03f;
        const float uy = 0.0f;
        
        float rho = (f[0 + node] + f[3 + node] + f[4 + node] + 
                    f[6 + node] + f[7 + node]) / (1.0f - ux - uy);
        
        // f1 (right) - from inlet condition
        f[1 + node] = f[3 + node] + (2.0f/3.0f) * rho * ux;
        
        // f2 (up) - from wall condition
        f[2 + node] = f[4 + node];
        
        // f5 (up-right) - combined effect
        f[5 + node] = f[7 + node] + (1.0f/6.0f) * rho * (ux + uy);
        
        // f8 (down-right) - from inlet
        f[8 + node] = f[6 + node] + (1.0f/6.0f) * rho * (ux - uy);
    }
    
    __device__ static void apply_bottom_left(float* f, float* f_back, const int* C, const int* OPP, int node) {
        node = get_node_index(node);
        
        const float ux = 0.03f;
        const float uy = 0.0f;
        
        float rho = (f[0 + node] + f[2 + node] + f[3 + node] + 
                    f[5 + node] + f[6 + node]) / (1.0f - ux + uy);
        
        // f1 (right) - from inlet condition
        f[1 + node] = f[3 + node] + (2.0f/3.0f) * rho * ux;
        
        // f4 (down) - from wall condition
        f[4 + node] = f[2 + node];
        
        // f5 (up-right) - from inlet
        f[5 + node] = f[7 + node] + (1.0f/6.0f) * rho * (ux + uy);
        
        // f8 (down-right) - combined effect
        f[8 + node] = f[6 + node] + (1.0f/6.0f) * rho * (ux - uy);
    }
};

#endif // ! CORNER_H
