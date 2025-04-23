#ifndef BOUNCE_BACK_BOUNDARY_H
#define BOUNCE_BACK_BOUNDARY_H

#include "core/lbm_constants.cuh"


// TODO: I could probably just ignore the x and y part, and just make separate cases for top
// and bottom boundaries, one for left/right boundaries and get rid of the loop and branching
// I am going to redo this when we start dealing with domain boundaries in separate kernels.

template <int dim>
struct BounceBack;

template <>
struct BounceBack<2> {
    bool x, y;
    
    __host__ __device__
    BounceBack(bool x=true, bool y=true) : x(x), y(y) {}
    
    __device__
    inline void apply(float* f, float* f_back, int node) {
        const int x = node % NX;
        const int y = node / NX;
        
        // float saved[9];
        // for (int j = 0; j < 9; j++) {
        //     saved[j] = f[get_node_index(node, j)];
        // }
        
        for (int i = 1; i < 9; i++) {
            int x_neigh = x + C[2 * i];
            int y_neigh = y + C[2 * i + 1];
            
            bool is_x_boundary = this->x && (x_neigh < 0 || x_neigh >= NX);
            bool is_y_boundary = this->y && (y_neigh < 0 || y_neigh >= NY);
            
            if (is_x_boundary || is_y_boundary) {
                int opp_i = OPP[i];
                float f_value = f[get_node_index(node, i)];
                f[get_node_index(node, opp_i)] = f_value;
                
                if (f_value < 0.0f) {
                    printf("[BOUNCE_BACK_2D] Bounced negative value at node (%d, %d): from dir %d to dir %d, value = %f\n", 
                           x, y, i, opp_i, f_value);
                }
            }
        }
    }
};

template <>
struct BounceBack<3> {
    bool x, y, z;
    
    __host__ __device__
    BounceBack(bool x=true, bool y=true, bool z=true) : x(x), y(y), z(z) {}
    
    __device__
    inline void apply(float* f, float* f_back, int node) {
        int x, y, z;
        get_coords_from_node(node, x, y, z);
        
        // float saved[27];
        // for (int j = 0; j < 27; j++) {
        //     saved[j] = f[get_node_index(node, j)];
        // }
        
        for (int i = 1; i < 27; i++) {
            int x_neigh = x + C[3 * i];
            int y_neigh = y + C[3 * i + 1];
            int z_neigh = z + C[3 * i + 2];
            
            bool is_x_boundary = this->x && (x_neigh < 0 || x_neigh >= NX);
            bool is_y_boundary = this->y && (y_neigh < 0 || y_neigh >= NY);
            bool is_z_boundary = this->z && (z_neigh < 0 || z_neigh >= NZ);
            
            if (is_x_boundary || is_y_boundary || is_z_boundary) {
                int opp_i = OPP[i];
                float f_value = f[get_node_index(node, i)];
                f[get_node_index(node, opp_i)] = f_value;
                // printf("I am not insane");
                
                if (f_value < 0.0f) {
                    printf("[BOUNCE_BACK_3D] Bounced negative value at node (%d, %d, %d): from dir %d to dir %d, value = %f\n", 
                           x, y, z, i, opp_i, f_value);
                }
            }
        }
    }
};
#endif // ! BOUNCE_BACK_BOUNDARY_H
