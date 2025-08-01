#include "core/lbm_constants.cuh"
#include "util/utility.cuh"

struct ConvectiveOutletX {
    __device__ static void apply(float* f, int node) {
        int x,y,z; get_coords_from_node(node,x,y,z);
        int node1 = get_node_from_coords(x-1,y,z);
        #pragma unroll
        for(int q=0;q<quadratures;q++){
            if(C[3*q+0]<0){            // outgoing cx <0
              f[get_node_index(node,q)] = f[get_node_index(node1,q)];
            }
        }
    }
};
