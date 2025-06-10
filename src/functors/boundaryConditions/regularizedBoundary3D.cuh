#ifndef REGULARIZED_BOUNDARY_H
#define REGULARIZED_BOUNDARY_H

#include "core/lbm_constants.cuh"

// Important to note that this is domain only. Given that, inlets and no-slip boundaries just differ from the expected velocity at the
// boudary (e.g. inlet left for a velocity with dir (1,0,0) is RegularizedBounceBack::apply(f, node, u_x, 0, 0) on a node at x=0). We still hold problems with corners, but that's for another time.
struct RegularizedBoundary {

    __device__ static inline void apply(float* f, int node, float wall_ux, float wall_uy, float wall_uz) {
        int x, y, z;
        get_coords_from_node(node, x, y, z);

        int normal_dim = -1;
        int normal_dir = 0; // +1 or -1

        if (x == 0) { normal_dim = 0; normal_dir = 1; }
        else if (x == NX - 1) { normal_dim = 0; normal_dir = -1; }
        else if (y == 0) { normal_dim = 1; normal_dir = 1; }
        else if (y == NY - 1) { normal_dim = 1; normal_dir = -1; }
        else if (z == 0) { normal_dim = 2; normal_dir = 1; }
        else if (z == NZ - 1) { normal_dim = 2; normal_dir = -1; }

        if (normal_dim == -1) return;

        float rho = compute_density(f, node, normal_dim, normal_dir, wall_ux, wall_uy, wall_uz);

        float f_eq[quadratures];
        compute_equilibrium(f_eq, rho, wall_ux, wall_uy, wall_uz);

        float Pi[6]; // xx, yy, zz, xy, xz, yz
        compute_stress_tensor(Pi, f, f_eq, node, normal_dim, normal_dir);

        regularize_all_populations(f, node, f_eq, Pi);
    }

private:
    __device__ static inline float compute_density(float* f, int node, int normal_dim, int normal_dir, float ux, float uy, float uz) {
        float rho_parallel = 0.0f;
        float rho_incoming = 0.0f;

        for (int i = 0; i < quadratures; ++i) {
            int c_normal = C[i * dimensions + normal_dim];

            if (c_normal * normal_dir < 0) {
                rho_incoming += f[get_node_index(node, i)];
            } else if (c_normal == 0) {
                rho_parallel += f[get_node_index(node, i)];
            }
        }
        
        float u_norm = (normal_dim == 0) ? ux : (normal_dim == 1) ? uy : uz;

        if (abs(1.0f - u_norm) < 1e-6) {
            return 1.0f;
        }
        
        return (rho_parallel + 2.0f * rho_incoming) / (1.0f - u_norm);
    }

    __device__ static inline void compute_equilibrium(float f_eq[quadratures], float rho, float ux, float uy, float uz) {

        for (int i = 0; i < quadratures; ++i) {
            float c_dot_u = C[i * dimensions + 0] * ux + C[i * dimensions + 1] * uy + C[i * dimensions + 2] * uz;
            float u_dot_u = ux * ux + uy * uy + uz * uz;
            f_eq[i] = WEIGHTS[i] * rho * (1.0f + c_dot_u / cs2 + (c_dot_u * c_dot_u) / (2.0f * cs2 * cs2) - u_dot_u / (2.0f * cs2));
        }

    }

    __device__ static inline void compute_stress_tensor(float Pi[6], float* f, const float f_eq[quadratures], int node, int normal_dim, int normal_dir) {
        Pi[0] = Pi[1] = Pi[2] = Pi[3] = Pi[4] = Pi[5] = 0.0f;

        for (int i = 0; i < quadratures; ++i) {
            float f_neq_i;
            int c_norm = C[i * dimensions + normal_dim];

            if (c_norm * normal_dir > 0) {
                int opp = OPP[i];
                f_neq_i = (f[get_node_index(node, opp)] - f_eq[opp]);
            }
            else {
                f_neq_i = f[get_node_index(node, i)] - f_eq[i];
            }

            float cx = C[i * dimensions + 0];
            float cy = C[i * dimensions + 1];
            float cz = C[i * dimensions + 2];

            Pi[0] += cx * cx * f_neq_i; // Pi_xx
            Pi[1] += cy * cy * f_neq_i; // Pi_yy
            Pi[2] += cz * cz * f_neq_i; // Pi_zz
            Pi[3] += cx * cy * f_neq_i; // Pi_xy
            Pi[4] += cx * cz * f_neq_i; // Pi_xz
            Pi[5] += cy * cz * f_neq_i; // Pi_yz
        }
    }

    __device__ static inline void regularize_all_populations(float* f, int node, const float f_eq[quadratures], const float Pi[6]) {

        for (int i = 0; i < quadratures; ++i) {
            float cx = C[i * dimensions + 0];
            float cy = C[i * dimensions + 1];
            float cz = C[i * dimensions + 2];

            float Q_xx = cx * cx - cs2;
            float Q_yy = cy * cy - cs2;
            float Q_zz = cz * cz - cs2;
            float Q_xy = cx * cy;
            float Q_xz = cx * cz;
            float Q_yz = cy * cz;

            float f_neq = (WEIGHTS[i] / (2.0f * cs2 * cs2)) * (
                Q_xx * Pi[0] + Q_yy * Pi[1] + Q_zz * Pi[2] +
                2.0f * (Q_xy * Pi[3] + Q_xz * Pi[4] + Q_yz * Pi[5])
            );

            f[get_node_index(node, i)] = f_eq[i] + f_neq;
        }
    }
};

#endif // !REGULARIZED_BOUNDARY_H
