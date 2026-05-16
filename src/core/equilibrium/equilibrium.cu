#include "core/equilibrium/equilibrium.cuh"

template<>
__device__
void LBM<2>::equilibrium_node_second_order(float* f_eq, float ux, float uy, float uz, float rho, int node) {
    float u_dot_u = ux * ux + uy * uy;
    float cs  = 1.0f / sqrt(3.0f);
    float cs2 = cs * cs;
    float cs4 = cs2 * cs2;


    f_eq[get_node_index(node, 0)] = WEIGHTS[0]*rho*
        (1 + 0.5*pow(C[0]*ux + C[1]*uy, 2)/cs4 - 0.5*u_dot_u/cs2 + 1.0*(C[0]*ux + C[1]*uy)/cs2);

    f_eq[get_node_index(node, 1)] = WEIGHTS[1]*rho*
        (1 + 0.5*pow(C[2]*ux + C[3]*uy, 2)/cs4 - 0.5*u_dot_u/cs2 + 1.0*(C[2]*ux + C[3]*uy)/cs2);

    f_eq[get_node_index(node, 2)] = WEIGHTS[2]*rho*
        (1 + 0.5*pow(C[4]*ux + C[5]*uy, 2)/cs4 - 0.5*u_dot_u/cs2 + 1.0*(C[4]*ux + C[5]*uy)/cs2);

    f_eq[get_node_index(node, 3)] = WEIGHTS[3]*rho*
        (1 + 0.5*pow(C[6]*ux + C[7]*uy, 2)/cs4 - 0.5*u_dot_u/cs2 + 1.0*(C[6]*ux + C[7]*uy)/cs2);

    f_eq[get_node_index(node, 4)] = WEIGHTS[4]*rho*
        (1 + 0.5*pow(C[8]*ux + C[9]*uy, 2)/cs4 - 0.5*u_dot_u/cs2 + 1.0*(C[8]*ux + C[9]*uy)/cs2);

    f_eq[get_node_index(node, 5)] = WEIGHTS[5]*rho*
        (1 + 0.5*pow(C[10]*ux + C[11]*uy, 2)/cs4 - 0.5*u_dot_u/cs2 + 1.0*(C[10]*ux + C[11]*uy)/cs2);

    f_eq[get_node_index(node, 6)] = WEIGHTS[6]*rho*
        (1 + 0.5*pow(C[12]*ux + C[13]*uy, 2)/cs4 - 0.5*u_dot_u/cs2 + 1.0*(C[12]*ux + C[13]*uy)/cs2);

    f_eq[get_node_index(node, 7)] = WEIGHTS[7]*rho*
        (1 + 0.5*pow(C[14]*ux + C[15]*uy, 2)/cs4 - 0.5*u_dot_u/cs2 + 1.0*(C[14]*ux + C[15]*uy)/cs2);

    f_eq[get_node_index(node, 8)] = WEIGHTS[8]*rho*
        (1 + 0.5*pow(C[16]*ux + C[17]*uy, 2)/cs4 - 0.5*u_dot_u/cs2 + 1.0*(C[16]*ux + C[17]*uy)/cs2);

}

template<>
__device__
void LBM<3>::equilibrium_node_second_order(float* f_eq, float ux, float uy, float uz, float rho, int node) {
    // Canonical second-order Hermite equilibrium (BGK/MRT path).
    // This avoids expensive pow() expressions and keeps the model at O(u^2).
    constexpr float cs2 = 1.0f / 3.0f;
    constexpr float inv_cs2 = 3.0f;
    constexpr float inv_2cs4 = 4.5f;  // 1/(2*cs^4)
    constexpr float inv_2cs2 = 1.5f;  // 1/(2*cs^2)

    const float u_dot_u = ux * ux + uy * uy + uz * uz;
    const float common = 1.0f - inv_2cs2 * u_dot_u;

    #pragma unroll
    for (int q = 0; q < quadratures; ++q) {
        const float cx = C[3 * q];
        const float cy = C[3 * q + 1];
        const float cz = C[3 * q + 2];
        const float cu = cx * ux + cy * uy + cz * uz;
        const int idx = get_node_index(node, q);

        f_eq[idx] = WEIGHTS[q] * rho *
                    (common + inv_cs2 * cu + inv_2cs4 * cu * cu);
    }

    // debug_equilibrium(f_eq, node);

}

template<>
__device__
void LBM<2>::equilibrium_node(float* f_eq, float ux, float uy, float uz, float rho, int node) {
    equilibrium_node_second_order(f_eq, ux, uy, uz, rho, node);
}

template<>
__device__
void LBM<3>::equilibrium_node(float* f_eq, float ux, float uy, float uz, float rho, int node) {
    equilibrium_node_second_order(f_eq, ux, uy, uz, rho, node);
}
