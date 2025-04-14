#include "core/streaming/streaming.cuh"

template<>
__device__
void LBM<2>::stream_node(float* f, float* f_back, int node) {
    const int x = node % NX;
    const int y = node / NX;
    const int baseIdx = get_node_index(node, 0);

    for (int i=1; i < quadratures; i++) {
        int x_neigh = x + C[2*i];
        int y_neigh = y + C[2*i+1];

#ifdef PERIODIC_X
        x_neigh = (x_neigh + NX) % NX;
#endif
#ifdef PERIODIC_Y
        y_neigh = (y_neigh + NY) % NY;
#endif

        if (x_neigh < 0 || x_neigh >= NX || y_neigh < 0 || y_neigh >= NY) 
            continue;

        const int node_neigh = get_node_from_coords(x_neigh, y_neigh);
        const int idx_neigh = get_node_index(node_neigh, i);
        f_back[idx_neigh] = f[baseIdx + i];

        if (fabsf(f_back[idx_neigh]) > VALUE_THRESHOLD || f_back[idx_neigh] < -0.01f) {
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
    
    const int baseIdx = get_node_index(node, 0);

    for (int i=1; i < quadratures; i++) {
        int x_neigh = x + C[3*i];
        int y_neigh = y + C[3*i+1];
        int z_neigh = z + C[3*i+2];

#ifdef PERIODIC_X
        x_neigh = (x_neigh + NX) % NX;
#endif
#ifdef PERIODIC_Y
        y_neigh = (y_neigh + NY) % NY;
#endif
#ifdef PERIODIC_Z
        z_neigh = (z_neigh + NZ) % NZ;
#endif

        if (x_neigh < 0 || x_neigh >= NX || y_neigh < 0 || y_neigh >= NY || z_neigh < 0 || z_neigh >= NZ)
            continue;


        const int node_neigh = get_node_from_coords(x_neigh, y_neigh, z_neigh);
        const int idx_neigh = get_node_index(node_neigh, i);
        f_back[idx_neigh] = f[baseIdx + i];

    }
}