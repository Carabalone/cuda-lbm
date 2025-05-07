#ifndef LBM_COLLISION_CM_H
#define LBM_COLLISION_CM_H

#include "core/lbm.cuh"
#include "core/lbm_constants.cuh"
#include "core/collision/adapters.cuh"

template <int dim, typename AdapterType = NoAdapter>
struct CM {
    __device__ __forceinline__ static
    void apply(float* f, float* f_eq, float* u, float* force, int node);

    __device__ __forceinline__ static
    void cm_matrix_inverse(float* M_inv, float ux, float uy);
};

template <typename AdapterType>
struct CM<2, AdapterType> {
    __device__ __forceinline__ static
    void apply(float* f, float* f_eq, float* u, float* force, int node);

    __device__ __forceinline__ static
    void cm_matrix_inverse(float* M_inv, float ux, float uy);
};


template <typename AdapterType>
__device__ __forceinline__
void CM<2, AdapterType>::apply(float* f, float* f_eq, float* u, float* force, int node) {
    float ux = u[get_vec_index(node, 0)];
    float uy = u[get_vec_index(node, 1)];

    float Fx = force[get_vec_index(node, 0)];
    float Fy = force[get_vec_index(node, 1)];

    float rho = 0.0f;
    float old_f[quadratures]; // for debug

    for (int i = 0; i < quadratures; i++) {
        rho += f[get_node_index(node, i)];
        old_f[i] = f[get_node_index(node, i)];
    }
    
    float k[quadratures]      = {0.0f};
    float k_eq[quadratures]   = {0.0f};
    float k_post[quadratures] = {0.0f};
    float pi[3] = {0.0f}; // pi_xx, pi_xy, pi_yy

    float T_inv[quadratures * quadratures] = {0};
    
    k[0] = rho;    
    
    for (int i = 0; i < quadratures; i++) {
        float c_cx = C[2*i] - ux;
        float c_cy = C[2*i+1] - uy;
        float c_cx2 = c_cx * c_cx;
        float c_cy2 = c_cy * c_cy;
        float f_i = f[get_node_index(node, i)];
        
        k[1] += f_i * (c_cx);
        k[2] += f_i * (c_cy);
        k[3] += f_i * (c_cx2 + c_cy2);
        k[4] += f_i * (c_cx2 - c_cy2);
        k[5] += f_i * c_cx * c_cy;
        k[6] += f_i * c_cx2 * c_cy;
        k[7] += f_i * c_cx * c_cy2;
        k[8] += f_i * c_cx2 * c_cy2;

        pi[0] += f_i * C[2*i] * C[2*i];        
        pi[1] += f_i * C[2*i] * C[2*i+1];      
        pi[2] += f_i * C[2*i+1] * C[2*i+1];    
    }

    float pi_mag = sqrtf(pi[0] * pi[0] + 2.0f * pi[1] * pi[1] + pi[2] * pi[2]);

    const float cs2 = 1.0f / 3.0f;

    k_eq[0] = rho;
    k_eq[1] = 0.0f;
    k_eq[2] = 0.0f;
    k_eq[3] = 2.0f * rho * cs2;
    k_eq[4] = 0.0f;
    k_eq[5] = 0.0f;
    k_eq[6] = 0.0f;
    k_eq[7] = 0.0f;
    k_eq[8] = rho * cs2 * cs2;

    // De Rosis Universal 2019 appendix b.
    float F[quadratures] = {
        0.0f,
        Fx,
        Fy,
        0.0f,
        0.0f,
        0.0f,
        Fy * cs2,
        Fx * cs2,
        0.0f
    };

    float j_mag = sqrtf(ux*ux + uy*uy) * rho;

    float high_order_relaxation;
    if constexpr (!std::is_same_v<AdapterType, NoAdapter>) {
        high_order_relaxation = AdapterType::compute_higher_order_relaxation(
            rho, j_mag, pi_mag, d_moment_avg);
    }
    
    for (int i = 0; i < quadratures; i++) {
        float relaxation_rate;
        if constexpr (!std::is_same_v<AdapterType, NoAdapter>) {
            relaxation_rate = AdapterType::is_higher_order(i) ? high_order_relaxation : S[i];
        } else {
            relaxation_rate = S[i];
        };

        k_post[i] = k[i] - relaxation_rate * (k[i] - k_eq[i]) +  
                    (1.0f - 0.5f*relaxation_rate) * F[i];
    }
    
    cm_matrix_inverse(T_inv, ux, uy);

    for (int i = 0; i < quadratures; i++) {
        int idx = get_node_index(node, i);
        f[idx] = 0.0f;
        for (int j = 0; j < quadratures; j++) {
            f[idx] += T_inv[i*quadratures + j] * k_post[j];
        }

        if (fabsf(f[idx]) > VALUE_THRESHOLD || f[idx] < -0.01f) {
            int x, y, z;
            get_coords_from_node(node, x, y, z);
            
            printf("[WARNING][CM::apply] Node %d (x=%d,y=%d), Dir %d: f[%d] = %f → %f\n",
                  node, x, y, i, idx, old_f[i], f[idx]);
        }
    }
}

// Matrix inverse implementation
template <typename AdapterType>
__device__ __forceinline__
void CM<2, AdapterType>::cm_matrix_inverse(float* M_inv, float ux, float uy) {
    // Priority subexpressions
    float ux2 = ux*ux;
    float uy2 = uy*uy;
    float uxuy = ux*uy;
    float ux3 = ux*ux*ux;
    float uy3 = uy*uy*uy;
    float ux2uy = uy*(ux*ux);
    float uxuy2 = ux*(uy*uy);
    float ux2uy2 = (ux*ux)*(uy*uy);

    // Additional common subexpressions
    float x3 = -uxuy2;
    float x4 = ux + x3;
    float x5 = -uy;
    float x6 = ux2uy + x5;


    M_inv[0 * 9 + 0] = -ux2 + ux2uy2 - uy2 + 1.0f;
    M_inv[0 * 9 + 1] = -2.0f*ux + 2.0f*uxuy2;
    M_inv[0 * 9 + 2] = 2.0f*ux2uy + 2.0f*x5;
    M_inv[0 * 9 + 3] = 0.5f*ux2 + 0.5f*uy2 - 1.0f;
    M_inv[0 * 9 + 4] = -0.5f*ux2 + 0.5f*uy2;
    M_inv[0 * 9 + 5] = 4.0f*uxuy;
    M_inv[0 * 9 + 6] = 2.0f*uy;
    M_inv[0 * 9 + 7] = 2.0f*ux;
    M_inv[0 * 9 + 8] = 1.0f;

    M_inv[1 * 9 + 0] = 0.5f*ux + 0.5f*ux2 - 0.5f*ux2uy2 + 0.5f*x3;
    M_inv[1 * 9 + 1] = ux - 0.5f*uy2 + x3 + 0.5f;
    M_inv[1 * 9 + 2] = -ux2uy - uxuy;
    M_inv[1 * 9 + 3] = -0.25f*ux - 0.25f*ux2 - 0.25f*uy2 + 0.25f;
    M_inv[1 * 9 + 4] = 0.25f*ux + 0.25f*ux2 - 0.25f*uy2 + 0.25f;
    M_inv[1 * 9 + 5] = -2.0f*uxuy + x5;
    M_inv[1 * 9 + 6] = x5;
    M_inv[1 * 9 + 7] = -ux - 0.5f;
    M_inv[1 * 9 + 8] = -0.5f;

    M_inv[2 * 9 + 0] = -0.5f*ux2uy - 0.5f*ux2uy2 + 0.5f*uy + 0.5f*uy2;
    M_inv[2 * 9 + 1] = -uxuy + x3;
    M_inv[2 * 9 + 2] = -0.5f*ux2 - ux2uy + uy + 0.5f;
    M_inv[2 * 9 + 3] = -0.25f*ux2 - 0.25f*uy2 + 0.25f*x5 + 0.25f;
    M_inv[2 * 9 + 4] = 0.25f*ux2 - 0.25f*uy2 + 0.25f*x5 - 0.25f;
    M_inv[2 * 9 + 5] = -ux - 2.0f*uxuy;
    M_inv[2 * 9 + 6] = x5 - 0.5f;
    M_inv[2 * 9 + 7] = -ux;
    M_inv[2 * 9 + 8] = -0.5f;

    M_inv[3 * 9 + 0] = -0.5f*ux + 0.5f*ux2 - 0.5f*ux2uy2 + 0.5f*uxuy2;
    M_inv[3 * 9 + 1] = ux + 0.5f*uy2 + x3 - 0.5f;
    M_inv[3 * 9 + 2] = -ux2uy + uxuy;
    M_inv[3 * 9 + 3] = 0.25f*ux - 0.25f*ux2 - 0.25f*uy2 + 0.25f;
    M_inv[3 * 9 + 4] = -0.25f*ux + 0.25f*ux2 - 0.25f*uy2 + 0.25f;
    M_inv[3 * 9 + 5] = -2.0f*uxuy + uy;
    M_inv[3 * 9 + 6] = x5;
    M_inv[3 * 9 + 7] = 0.5f - ux;
    M_inv[3 * 9 + 8] = -0.5f;

    M_inv[4 * 9 + 0] = 0.5f*ux2uy - 0.5f*ux2uy2 + 0.5f*uy2 + 0.5f*x5;
    M_inv[4 * 9 + 1] = uxuy + x3;
    M_inv[4 * 9 + 2] = 0.5f*ux2 - ux2uy + uy - 0.5f;
    M_inv[4 * 9 + 3] = -0.25f*ux2 + 0.25f*uy - 0.25f*uy2 + 0.25f;
    M_inv[4 * 9 + 4] = 0.25f*ux2 + 0.25f*uy - 0.25f*uy2 - 0.25f;
    M_inv[4 * 9 + 5] = ux - 2.0f*uxuy;
    M_inv[4 * 9 + 6] = x5 + 0.5f;
    M_inv[4 * 9 + 7] = -ux;
    M_inv[4 * 9 + 8] = -0.5f;

    M_inv[5 * 9 + 0] = 0.25f*ux2uy + 0.25f*ux2uy2 + 0.25f*uxuy + 0.25f*uxuy2;
    M_inv[5 * 9 + 1] = 0.5f*uxuy + 0.5f*uxuy2 + 0.25f*uy + 0.25f*uy2;
    M_inv[5 * 9 + 2] = 0.25f*ux + 0.25f*ux2 + 0.5f*ux2uy + 0.5f*uxuy;
    M_inv[5 * 9 + 3] = 0.125f*ux + 0.125f*ux2 + 0.125f*uy + 0.125f*uy2;
    M_inv[5 * 9 + 4] = -0.125f*ux - 0.125f*ux2 + 0.125f*uy + 0.125f*uy2;
    M_inv[5 * 9 + 5] = 0.5f*ux + uxuy + 0.5f*uy + 0.25f;
    M_inv[5 * 9 + 6] = 0.5f*uy + 0.25f;
    M_inv[5 * 9 + 7] = 0.5f*ux + 0.25f;
    M_inv[5 * 9 + 8] = 0.25f;

    M_inv[6 * 9 + 0] = 0.25f*ux2uy + 0.25f*ux2uy2 - 0.25f*uxuy + 0.25f*x3;
    M_inv[6 * 9 + 1] = 0.5f*uxuy + 0.5f*uxuy2 - 0.25f*uy2 + 0.25f*x5;
    M_inv[6 * 9 + 2] = -0.25f*ux + 0.25f*ux2 + 0.5f*ux2uy - 0.5f*uxuy;
    M_inv[6 * 9 + 3] = -0.125f*ux + 0.125f*ux2 + 0.125f*uy + 0.125f*uy2;
    M_inv[6 * 9 + 4] = 0.125f*ux - 0.125f*ux2 + 0.125f*uy + 0.125f*uy2;
    M_inv[6 * 9 + 5] = 0.5f*ux + uxuy + 0.5f*x5 - 0.25f;
    M_inv[6 * 9 + 6] = 0.5f*uy + 0.25f;
    M_inv[6 * 9 + 7] = 0.5f*ux - 0.25f;
    M_inv[6 * 9 + 8] = 0.25f;

    M_inv[7 * 9 + 0] = -0.25f*ux2uy + 0.25f*ux2uy2 + 0.25f*uxuy + 0.25f*x3;
    M_inv[7 * 9 + 1] = -0.5f*uxuy + 0.5f*uxuy2 + 0.25f*uy - 0.25f*uy2;
    M_inv[7 * 9 + 2] = 0.25f*ux - 0.25f*ux2 + 0.5f*ux2uy - 0.5f*uxuy;
    M_inv[7 * 9 + 3] = -0.125f*ux + 0.125f*ux2 + 0.125f*uy2 + 0.125f*x5;
    M_inv[7 * 9 + 4] = 0.125f*ux - 0.125f*ux2 + 0.125f*uy2 + 0.125f*x5;
    M_inv[7 * 9 + 5] = -0.5f*ux + uxuy + 0.5f*x5 + 0.25f;
    M_inv[7 * 9 + 6] = 0.5f*uy - 0.25f;
    M_inv[7 * 9 + 7] = 0.5f*ux - 0.25f;
    M_inv[7 * 9 + 8] = 0.25f;

    M_inv[8 * 9 + 0] = -0.25f*ux2uy + 0.25f*ux2uy2 - 0.25f*uxuy + 0.25f*uxuy2;
    M_inv[8 * 9 + 1] = -0.5f*uxuy + 0.5f*uxuy2 + 0.25f*uy2 + 0.25f*x5;
    M_inv[8 * 9 + 2] = -0.25f*ux - 0.25f*ux2 + 0.5f*ux2uy + 0.5f*uxuy;
    M_inv[8 * 9 + 3] = 0.125f*ux + 0.125f*ux2 + 0.125f*uy2 + 0.125f*x5;
    M_inv[8 * 9 + 4] = -0.125f*ux - 0.125f*ux2 + 0.125f*uy2 + 0.125f*x5;
    M_inv[8 * 9 + 5] = -0.5f*ux + uxuy + 0.5f*uy - 0.25f;
    M_inv[8 * 9 + 6] = 0.5f*uy - 0.25f;
    M_inv[8 * 9 + 7] = 0.5f*ux + 0.25f;
    M_inv[8 * 9 + 8] = 0.25f;
}

#include "CM_D3Q27.cuh"

#endif // ! LBM_COLLISION_CM_H