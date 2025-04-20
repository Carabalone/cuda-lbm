#ifndef LBM_COLLISION_BGK_H
#define LBM_COLLISION_BGK_H

#include "core/lbm.cuh"
#include "core/lbm_constants.cuh"

template <int dim>
struct BGK {
    __device__ __forceinline__ static
    void apply(float* f, float* f_eq, float* u, float* force, int node);
};

template<>
__device__ __forceinline__
void BGK<2>::apply(float* f, float* f_eq, float* u, float* force, int node) {
    for (int q = 0; q < quadratures; q++) {
        int idx = get_node_index(node, q);
        float cx  = C[dimensions*q];
        float cy  = C[dimensions*q+1];
        float fx  = force[get_vec_index(node, 0)];
        float fy  = force[get_vec_index(node, 1)];
        float ux  = u[get_vec_index(node, 0)];
        float uy  = u[get_vec_index(node, 1)];
        float cs2 = 1.0f / 3.0f;
        
        float force_term = WEIGHTS[q] * (
            (1.0f - 0.5f * omega) * (
                (cx - ux) / cs2 + 
                (cx * ux + cy * uy) * cx / (cs2 * cs2)
            ) * fx +
            (1.0f - 0.5f * omega) * (
                (cy - uy) / cs2 + 
                (cx * ux + cy * uy) * cy / (cs2 * cs2)
            ) * fy
        );

        float old_val = f[idx];
        float new_val = f[idx] - omega * (f[idx] - f_eq[idx]) + force_term;
        f[idx] = new_val;

        if (fabsf(f[idx]) > VALUE_THRESHOLD || f[idx] < -0.01f) {
            int x, y, z;
            get_coords_from_node(node, x, y, z);

            printf("[WARNING][BGK::apply] Node %d (x=%d,y=%d), Dir %d, idx=%d: f[%d] = %f - %f*(%f - %f) = %f\n\tforces: (%.4f, %.4f)/ft: %.4f\n",
                   node, x, y, q, idx,
                   idx, old_val, omega, old_val, f_eq[idx], new_val, fx, fy, force_term);
        }
    }
}

template<>
__device__ __forceinline__
void BGK<3>::apply(float* f, float* f_eq, float* u, float* force, int node) {
    for (int q = 0; q < quadratures; q++) {
        int idx = get_node_index(node, q);
        float cx  = C[dimensions*q];
        float cy  = C[dimensions*q+1];
        float cz  = C[dimensions*q+2];
        float fx  = force[get_vec_index(node, 0)];
        float fy  = force[get_vec_index(node, 1)];
        float fz  = force[get_vec_index(node, 2)];
        float ux  = u[get_vec_index(node, 0)];
        float uy  = u[get_vec_index(node, 1)];
        float uz  = u[get_vec_index(node, 2)];
        float cs2 = 1.0f / 3.0f;
        
        float u_dot_c = cx * ux + cy * uy + cz * uz;
        
        float force_term = WEIGHTS[q] * (
            (1.0f - 0.5f * omega) * (
                (cx - ux) / cs2 + u_dot_c * cx / (cs2 * cs2)
            ) * fx +
            (1.0f - 0.5f * omega) * (
                (cy - uy) / cs2 + u_dot_c * cy / (cs2 * cs2)
            ) * fy +
            (1.0f - 0.5f * omega) * (
                (cz - uz) / cs2 + u_dot_c * cz / (cs2 * cs2)
            ) * fz
        );

        float old_val = f[idx];
        float new_val = f[idx] - omega * (f[idx] - f_eq[idx]) + force_term;
        // if (node == 6403) {
        //     printf("idx=%d | f[%d] | f_new[%d] = {%.4f, %.4f}\n", idx, q, q, old_val, new_val);
        // }
        f[idx] = new_val;

        if (fabsf(f[idx]) > VALUE_THRESHOLD || f[idx] < -0.01f) {
            int x, y, z;
            get_coords_from_node(node, x, y, z);

            printf("[WARNING][BGK::apply] Node %d (x=%d,y=%d,z=%d), Dir %d, idx=%d: f[%d] = %f - %f*(%f - %f) = %f\n\tforces: (%.4f, %.4f, %.4f)/ft: %.4f\n",
                   node, x, y, z, q, idx,
                   idx, old_val, omega, old_val, f_eq[idx], new_val, fx, fy, fz, force_term);
        }
    }
}

#endif // ! LBM_COLLISION_BGK_H