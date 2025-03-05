#ifndef ZOU_HE_H
#define ZOU_HE_H

#include "lbm_constants.cuh"

struct ZouHe {

    __device__
    static void apply_left(float* f, float* f_back, float u_lid, int node) {

        node = get_node_index(node);
        const float ux = 0.03f;
        const float uy = 0.0f;

        float rho = (f[0 + node] + f[2 + node] + f[3 + node] + 
                    f[4 + node] + f[6 + node] + f[7 + node]) / (1.0f - ux);

        f[1 + node] = f[3 + node] + (2.0f/3.0f) * rho * ux;

        float f2_f4_diff = f[2 + node] - f[4 + node];
        
        float f57_diff =  f2_f4_diff / 2.0f;
        float f68_diff = -f2_f4_diff / 2.0f;
        
        f[5 + node] = f[7 + node] + f57_diff + (1.0f/6.0f) * rho * ux;
        f[8 + node] = f[6 + node] + f68_diff + (1.0f/6.0f) * rho * ux;

    }

    __device__
    static void apply_top(float* f, float* f_back, float u_lid, int node) {
        const int x = node % NX;
        const int y = node / NX;
        node = get_node_index(node);
        const float ux = u_lid;
        const float uy = 0.0f;

        float rho = (f[0 + node] + f[1 + node] + f[3 + node] + 
                    2.0f * (f[2 + node] + f[5 + node] + f[6 + node])) / (1.0f + uy);

        f[4 + node] = f[2 + node] - (2.0f/3.0f) * rho * uy;
        
        float f1_f3_diff = f[1 + node] - f[3 + node];

        f[7 + node] = f[5 + node] + 0.5f * f1_f3_diff - (1.0f/2.0f) * rho * ux - (1.0f/6.0f) * rho * uy;
        f[8 + node] = f[6 + node] - 0.5f * f1_f3_diff + (1.0f/2.0f) * rho * ux - (1.0f/6.0f) * rho * uy;

        // if (rho < 0.0f) {
        //     printf("Negative rho detected at top boundary, node %d: %f\n", node, rho);
        // }
        // if (f[4 + node] < 0.0f || f[7 + node] < 0.0f || f[8 + node] < 0.0f) {
        //     printf(
        //         "Issue detected at top boundary, node (%d, %d):\n"
        //         "  Negative rho: %s (rho=%.4f)\n"
        //         "  Original distributions: f0=%.4f f1=%.4f f2=%.4f f3=%.4f f4=%.4f f5=%.4f f6=%.4f f7=%.4f f8=%.4f\n"
        //         "  Updated distributions: f4=%.4f f7=%.4f f8=%.4f\n",
        //         x, y,
        //         (rho < 0.0f) ? "Yes" : "No", rho,
        //         f_original[0], f_original[1], f_original[2], f_original[3], f_original[4], 
        //         f_original[5], f_original[6], f_original[7], f_original[8],
        //         f[4 + node], f[7 + node], f[8 + node]
        //     );
        // }
    }
};

#endif // ! ZOU_HE_H
