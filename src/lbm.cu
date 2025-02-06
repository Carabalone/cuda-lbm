#include "lbm.cuh"

// u -> velocity, rho -> density
void LBM::equilibrium(float* f_eq, int node, float ux, float uy, float rho) {
    f_eq[get_node_index(i)]    =  rho * (ux * C[0]   +  uy * C[1]   +  1) * WEIGHTS[0];
    f_eq[get_node_index(i+1)]  =  rho * (ux * C[2]   +  uy * C[3]   +  1) * WEIGHTS[1];
    f_eq[get_node_index(i+2)]  =  rho * (ux * C[4]   +  uy * C[5]   +  1) * WEIGHTS[2];
    f_eq[get_node_index(i+3)]  =  rho * (ux * C[6]   +  uy * C[7]   +  1) * WEIGHTS[3];
    f_eq[get_node_index(i+4)]  =  rho * (ux * C[8]   +  uy * C[9]   +  1) * WEIGHTS[4];
    f_eq[get_node_index(i+5)]  =  rho * (ux * C[10]  +  uy * C[11]  +  1) * WEIGHTS[5];
    f_eq[get_node_index(i+6)]  =  rho * (ux * C[12]  +  uy * C[13]  +  1) * WEIGHTS[6];
    f_eq[get_node_index(i+7)]  =  rho * (ux * C[14]  +  uy * C[15]  +  1) * WEIGHTS[7];
    f_eq[get_node_index(i+8)]  =  rho * (ux * C[16]  +  uy * C[17]  +  1) * WEIGHTS[8];
}

void LBM::init(float* f, float* f_back, float* f_eq, float* rho, float* u) {

    cudaMemset(rho, 1.0f, sizeof(float) * NX * NY);

    // equilibrium(f_eq)
}
