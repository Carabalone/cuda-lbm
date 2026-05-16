#include "core/streaming/streaming.cuh"

template<>
__device__
void LBM<2>::stream_node(float* f, float* f_back, int node) {
    const int x = node % NX;
    const int y = node / NX;

    for (int i=1; i < quadratures; i++) {
        int x_src = x - C[2*i];
        int y_src = y - C[2*i+1];

#ifdef PERIODIC_X
        x_src = (x_src + NX) % NX;
#endif
#ifdef PERIODIC_Y
        y_src = (y_src + NY) % NY;
#endif

        if (x_src < 0 || x_src >= NX || y_src < 0 || y_src >= NY) 
            continue;

        const int node_src = get_node_from_coords(x_src, y_src);
        const int idx = get_node_index(node, i);
        f_back[idx] = f[get_node_index(node_src, i)];

        if (fabsf(f_back[idx]) > VALUE_THRESHOLD || f_back[idx] < -0.01f) {
            // printf("[WARNING][stream_node] Pushing negative/large value: "
            //     "Node (x=%3d, y=%3d) is pushing f[%d]=% .6f in Dir %d to neighbor at (x=%3d, y=%3d)\n",
            //     x, y, i, f[baseIdx + i], i, x_neigh, y_neigh);
        }
    }
}

template<>
__device__
void LBM<3>::stream_node(float* f, float* f_back, int node) {
    int x, y, z;
    get_coords_from_node(node, x, y, z);
    
    for (int i=1; i < quadratures; i++) {
        int x_src = x - C[3*i];
        int y_src = y - C[3*i+1];
        int z_src = z - C[3*i+2];

#ifdef PERIODIC_X
        x_src = (x_src + NX) % NX;
#endif
#ifdef PERIODIC_Y
        y_src = (y_src + NY) % NY;
#endif
#ifdef PERIODIC_Z
        z_src = (z_src + NZ) % NZ;
#endif

        if (x_src < 0 || x_src >= NX || y_src < 0 || y_src >= NY || z_src < 0 || z_src >= NZ) {
            continue;
        }


        const int node_src = get_node_from_coords(x_src, y_src, z_src);
        const int idx = get_node_index(node, i);
        f_back[idx] = f[get_node_index(node_src, i)];

    }
}