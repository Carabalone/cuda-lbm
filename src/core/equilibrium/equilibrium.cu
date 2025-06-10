#include "core/equilibrium/equilibrium.cuh"

template<>
__device__
void LBM<2>::equilibrium_node(float* f_eq, float ux, float uy, float uz, float rho, int node) {
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
void LBM<3>::equilibrium_node(float* f_eq, float ux, float uy, float uz, float rho, int node) {
    float ux2 = ux * ux;
    float uy2 = uy * uy;
    float uz2 = uz * uz;
    float u_dot_u = ux2 + uy2 + uz2;
    float cs2 = 1.0f/3.0f;
    float cs4 = cs2 * cs2;
    float two_cs4 = 2.0f * cs4;
    float cs2_cs4 = cs2 * cs4;
    float inv_cs2_cs4 = 1.0f / cs2_cs4;
    float two_cs2_cs4 = 2.0f * cs2 * cs4;
    float half_rho = 0.5f * rho;

    // Suga (2015)
    // f_eq[get_node_index(node, 0)] = -half_rho*WEIGHTS[0]*(-2.0f*cs2 + ux2 + uy2 + uz2)*1.0f/cs2;
    // f_eq[get_node_index(node, 1)] = inv_cs2_cs4*half_rho*WEIGHTS[1]*(1.0f*cs2*ux2 + two_cs2_cs4 + two_cs4*ux - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 2)] = inv_cs2_cs4*half_rho*WEIGHTS[2]*(1.0f*cs2*ux2 + two_cs2_cs4 - two_cs4*ux - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 3)] = inv_cs2_cs4*half_rho*WEIGHTS[3]*(1.0f*cs2*uy2 + two_cs2_cs4 + two_cs4*uy - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 4)] = inv_cs2_cs4*half_rho*WEIGHTS[4]*(1.0f*cs2*uy2 + two_cs2_cs4 - two_cs4*uy - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 5)] = inv_cs2_cs4*half_rho*WEIGHTS[5]*(1.0f*cs2*uz2 + two_cs2_cs4 + two_cs4*uz - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 6)] = inv_cs2_cs4*half_rho*WEIGHTS[6]*(1.0f*cs2*uz2 + two_cs2_cs4 - two_cs4*uz - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 7)] = inv_cs2_cs4*half_rho*WEIGHTS[7]*(1.0f*cs2*pow(ux + uy, 2) + two_cs2_cs4 + two_cs4*(ux + uy) - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 8)] = inv_cs2_cs4*half_rho*WEIGHTS[8]*(1.0f*cs2*pow(ux - uy, 2) + two_cs2_cs4 - two_cs4*(ux - uy) - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 9)] = inv_cs2_cs4*half_rho*WEIGHTS[9]*(1.0f*cs2*pow(ux + uy, 2) + two_cs2_cs4 - two_cs4*(ux + uy) - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 10)] = inv_cs2_cs4*half_rho*WEIGHTS[10]*(1.0f*cs2*pow(ux - uy, 2) + two_cs2_cs4 + two_cs4*(ux - uy) - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 11)] = inv_cs2_cs4*half_rho*WEIGHTS[11]*(1.0f*cs2*pow(ux + uz, 2) + two_cs2_cs4 + two_cs4*(ux + uz) - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 12)] = inv_cs2_cs4*half_rho*WEIGHTS[12]*(1.0f*cs2*pow(ux - uz, 2) + two_cs2_cs4 - two_cs4*(ux - uz) - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 13)] = inv_cs2_cs4*half_rho*WEIGHTS[13]*(1.0f*cs2*pow(ux + uz, 2) + two_cs2_cs4 - two_cs4*(ux + uz) - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 14)] = inv_cs2_cs4*half_rho*WEIGHTS[14]*(1.0f*cs2*pow(ux - uz, 2) + two_cs2_cs4 + two_cs4*(ux - uz) - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 15)] = inv_cs2_cs4*half_rho*WEIGHTS[15]*(1.0f*cs2*pow(uy + uz, 2) + two_cs2_cs4 + two_cs4*(uy + uz) - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 16)] = inv_cs2_cs4*half_rho*WEIGHTS[16]*(1.0f*cs2*pow(uy - uz, 2) + two_cs2_cs4 - two_cs4*(uy - uz) - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 17)] = inv_cs2_cs4*half_rho*WEIGHTS[17]*(1.0f*cs2*pow(uy + uz, 2) + two_cs2_cs4 - two_cs4*(uy + uz) - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 18)] = inv_cs2_cs4*half_rho*WEIGHTS[18]*(1.0f*cs2*pow(uy - uz, 2) + two_cs2_cs4 + two_cs4*(uy - uz) - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 19)] = inv_cs2_cs4*half_rho*WEIGHTS[19]*(1.0f*cs2*pow(ux + uy + uz, 2) + two_cs2_cs4 + two_cs4*(ux + uy + uz) - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 20)] = inv_cs2_cs4*half_rho*WEIGHTS[20]*(1.0f*cs2*pow(-ux + uy + uz, 2) + two_cs2_cs4 + two_cs4*(-ux + uy + uz) - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 21)] = inv_cs2_cs4*half_rho*WEIGHTS[21]*(1.0f*cs2*pow(ux + uy - uz, 2) + two_cs2_cs4 - two_cs4*(ux + uy - uz) - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 22)] = inv_cs2_cs4*half_rho*WEIGHTS[22]*(1.0f*cs2*pow(ux - uy + uz, 2) + two_cs2_cs4 + two_cs4*(ux - uy + uz) - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 23)] = inv_cs2_cs4*half_rho*WEIGHTS[23]*(1.0f*cs2*pow(ux + uy - uz, 2) + two_cs2_cs4 + two_cs4*(ux + uy - uz) - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 24)] = inv_cs2_cs4*half_rho*WEIGHTS[24]*(1.0f*cs2*pow(ux - uy + uz, 2) + two_cs2_cs4 - two_cs4*(ux - uy + uz) - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 25)] = inv_cs2_cs4*half_rho*WEIGHTS[25]*(1.0f*cs2*pow(ux + uy + uz, 2) + two_cs2_cs4 - two_cs4*(ux + uy + uz) - cs4*(ux2 + uy2 + uz2));
    // f_eq[get_node_index(node, 26)] = inv_cs2_cs4*half_rho*WEIGHTS[26]*(1.0f*cs2*pow(-ux + uy + uz, 2) + two_cs2_cs4 - two_cs4*(-ux + uy + uz) - cs4*(ux2 + uy2 + uz2));

    // De Rosis (2017)
    f_eq[get_node_index(node, 0)]  = half_rho*WEIGHTS[0]*(2.0f*cs2 - ux2 - uy2 - uz2)/cs2;
    f_eq[get_node_index(node, 1)]  = inv_cs2_cs4*half_rho*WEIGHTS[1]*(1.0f*cs2*ux2 + two_cs2_cs4 + two_cs4*ux - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 2)]  = inv_cs2_cs4*half_rho*WEIGHTS[2]*(1.0f*cs2*ux2 + two_cs2_cs4 - two_cs4*ux - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 3)]  = inv_cs2_cs4*half_rho*WEIGHTS[3]*(1.0f*cs2*uy2 + two_cs2_cs4 + two_cs4*uy - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 4)]  = inv_cs2_cs4*half_rho*WEIGHTS[4]*(1.0f*cs2*uy2 + two_cs2_cs4 - two_cs4*uy - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 5)]  = inv_cs2_cs4*half_rho*WEIGHTS[5]*(1.0f*cs2*uz2 + two_cs2_cs4 + two_cs4*uz - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 6)]  = inv_cs2_cs4*half_rho*WEIGHTS[6]*(1.0f*cs2*uz2 + two_cs2_cs4 - two_cs4*uz - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 7)]  = inv_cs2_cs4*half_rho*WEIGHTS[7]*(1.0f*cs2 *(ux + uy)*((ux + uy) + two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 8)]  = inv_cs2_cs4*half_rho*WEIGHTS[8]*(1.0f*cs2 *(ux - uy)*((ux - uy) - two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 9)]  = inv_cs2_cs4*half_rho*WEIGHTS[9]*(1.0f*cs2 *(ux - uy)*((ux - uy) + two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 10)] = inv_cs2_cs4*half_rho*WEIGHTS[10]*(1.0f*cs2*(ux + uy)*((ux + uy) - two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 11)] = inv_cs2_cs4*half_rho*WEIGHTS[11]*(1.0f*cs2*(ux + uz)*((ux + uz) + two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 12)] = inv_cs2_cs4*half_rho*WEIGHTS[12]*(1.0f*cs2*(ux - uz)*((ux - uz) - two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 13)] = inv_cs2_cs4*half_rho*WEIGHTS[13]*(1.0f*cs2*(ux - uz)*((ux - uz) + two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 14)] = inv_cs2_cs4*half_rho*WEIGHTS[14]*(1.0f*cs2*(ux + uz)*((ux + uz) - two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 15)] = inv_cs2_cs4*half_rho*WEIGHTS[15]*(1.0f*cs2*(uy + uz)*((uy + uz) + two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 16)] = inv_cs2_cs4*half_rho*WEIGHTS[16]*(1.0f*cs2*(uy - uz)*((uy - uz) - two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 17)] = inv_cs2_cs4*half_rho*WEIGHTS[17]*(1.0f*cs2*(uy - uz)*((uy - uz) + two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 18)] = inv_cs2_cs4*half_rho*WEIGHTS[18]*(1.0f*cs2*(uy + uz)*((uy + uz) - two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 19)] = inv_cs2_cs4*half_rho*WEIGHTS[19]*(1.0f*cs2*(ux + uy + uz)*((ux + uy + uz) + two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 20)] = inv_cs2_cs4*half_rho*WEIGHTS[20]*(1.0f*cs2*(-ux + uy + uz)*((-ux + uy + uz) + two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 21)] = inv_cs2_cs4*half_rho*WEIGHTS[21]*(1.0f*cs2*(ux - uy + uz)*((ux - uy + uz) + two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 22)] = inv_cs2_cs4*half_rho*WEIGHTS[22]*(1.0f*cs2*(ux + uy - uz)*((ux + uy - uz) - two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 23)] = inv_cs2_cs4*half_rho*WEIGHTS[23]*(1.0f*cs2*(ux + uy - uz)*((ux + uy - uz) + two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 24)] = inv_cs2_cs4*half_rho*WEIGHTS[24]*(1.0f*cs2*(ux - uy + uz)*((ux - uy + uz) - two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 25)] = inv_cs2_cs4*half_rho*WEIGHTS[25]*(1.0f*cs2*(-ux + uy + uz)*((-ux + uy + uz) - two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
    f_eq[get_node_index(node, 26)] = inv_cs2_cs4*half_rho*WEIGHTS[26]*(1.0f*cs2*(ux + uy + uz)*((ux + uy + uz) - two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));

    // debug_equilibrium(f_eq, node);

}


__device__
void debug_equilibrium(float* f_eq, int node) {
    if ((node == 0) || (node == NX*NY*NZ-1) || (node == NX*NY*NZ/2)) {
        printf("EQ Node %d: {\n"
               "  \"f_eq[0]\": %.6f,\n"
               "  \"f_eq[1]\": %.6f,\n"
               "  \"f_eq[2]\": %.6f,\n"
               "  \"f_eq[3]\": %.6f,\n"
               "  \"f_eq[4]\": %.6f,\n"
               "  \"f_eq[5]\": %.6f,\n"
               "  \"f_eq[6]\": %.6f,\n"
               "  \"f_eq[7]\": %.6f,\n"
               "  \"f_eq[8]\": %.6f,\n"
               "  \"f_eq[9]\": %.6f,\n"
               "  \"f_eq[10]\": %.6f,\n"
               "  \"f_eq[11]\": %.6f,\n"
               "  \"f_eq[12]\": %.6f,\n"
               "  \"f_eq[13]\": %.6f,\n"
               "  \"f_eq[14]\": %.6f,\n"
               "  \"f_eq[15]\": %.6f,\n"
               "  \"f_eq[16]\": %.6f,\n"
               "  \"f_eq[17]\": %.6f,\n"
               "  \"f_eq[18]\": %.6f,\n"
               "  \"f_eq[19]\": %.6f,\n"
               "  \"f_eq[20]\": %.6f,\n"
               "  \"f_eq[21]\": %.6f,\n"
               "  \"f_eq[22]\": %.6f,\n"
               "  \"f_eq[23]\": %.6f,\n"
               "  \"f_eq[24]\": %.6f,\n"
               "  \"f_eq[25]\": %.6f\n"
               "  \"f_eq[26]\": %.6f\n"
               "}\n",
               node,
               f_eq[get_node_index(node, 0)], f_eq[get_node_index(node, 1)],
               f_eq[get_node_index(node, 2)], f_eq[get_node_index(node, 3)],
               f_eq[get_node_index(node, 4)], f_eq[get_node_index(node, 5)],
               f_eq[get_node_index(node, 6)], f_eq[get_node_index(node, 7)],
               f_eq[get_node_index(node, 8)], f_eq[get_node_index(node, 9)],
               f_eq[get_node_index(node, 10)], f_eq[get_node_index(node, 11)],
               f_eq[get_node_index(node, 12)], f_eq[get_node_index(node, 13)],
               f_eq[get_node_index(node, 14)], f_eq[get_node_index(node, 15)],
               f_eq[get_node_index(node, 16)], f_eq[get_node_index(node, 17)],
               f_eq[get_node_index(node, 18)], f_eq[get_node_index(node, 19)],
               f_eq[get_node_index(node, 20)], f_eq[get_node_index(node, 21)],
               f_eq[get_node_index(node, 22)], f_eq[get_node_index(node, 23)],
               f_eq[get_node_index(node, 24)], f_eq[get_node_index(node, 25)],
               f_eq[get_node_index(node, 26)]);
    }
}
