#ifndef GUO_OUTLET_3D_H
#define GUO_OUTLET_3D_H

struct GuoPressureOutlet3D {
    __device__ static inline void apply(float* f, float* rho, float* u, int node) {
        int x, y, z;
        get_coords_from_node(node, x, y, z);
        int interior = get_node_from_coords(x - 1, y, z);
    
        // Copy all unknown populations from interior
        for (int i = 0; i < quadratures; ++i) {
            if (C[i * dimensions + 0] < 0) { // unknown (flowing out)
                f[get_node_index(node, i)] = f[get_node_index(interior, i)];
            }
        }        
    }
    
private:
    
    __device__
    static void de_rosis_eq(float* f_eq, float ux, float uy, float uz, float rho) {
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

        // De Rosis (2017)
        f_eq[0]  = half_rho*WEIGHTS[0]*(2.0f*cs2 - ux2 - uy2 - uz2)/cs2;
        f_eq[1]  = inv_cs2_cs4*half_rho*WEIGHTS[1]*(1.0f*cs2*ux2 + two_cs2_cs4 + two_cs4*ux - cs4*(ux2 + uy2 + uz2));
        f_eq[2]  = inv_cs2_cs4*half_rho*WEIGHTS[2]*(1.0f*cs2*ux2 + two_cs2_cs4 - two_cs4*ux - cs4*(ux2 + uy2 + uz2));
        f_eq[3]  = inv_cs2_cs4*half_rho*WEIGHTS[3]*(1.0f*cs2*uy2 + two_cs2_cs4 + two_cs4*uy - cs4*(ux2 + uy2 + uz2));
        f_eq[4]  = inv_cs2_cs4*half_rho*WEIGHTS[4]*(1.0f*cs2*uy2 + two_cs2_cs4 - two_cs4*uy - cs4*(ux2 + uy2 + uz2));
        f_eq[5]  = inv_cs2_cs4*half_rho*WEIGHTS[5]*(1.0f*cs2*uz2 + two_cs2_cs4 + two_cs4*uz - cs4*(ux2 + uy2 + uz2));
        f_eq[6]  = inv_cs2_cs4*half_rho*WEIGHTS[6]*(1.0f*cs2*uz2 + two_cs2_cs4 - two_cs4*uz - cs4*(ux2 + uy2 + uz2));
        f_eq[7]  = inv_cs2_cs4*half_rho*WEIGHTS[7]*(1.0f*cs2 *(ux + uy)*((ux + uy) + two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[8]  = inv_cs2_cs4*half_rho*WEIGHTS[8]*(1.0f*cs2 *(ux - uy)*((ux - uy) - two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[9]  = inv_cs2_cs4*half_rho*WEIGHTS[9]*(1.0f*cs2 *(ux - uy)*((ux - uy) + two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[10] = inv_cs2_cs4*half_rho*WEIGHTS[10]*(1.0f*cs2*(ux + uy)*((ux + uy) - two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[11] = inv_cs2_cs4*half_rho*WEIGHTS[11]*(1.0f*cs2*(ux + uz)*((ux + uz) + two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[12] = inv_cs2_cs4*half_rho*WEIGHTS[12]*(1.0f*cs2*(ux - uz)*((ux - uz) - two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[13] = inv_cs2_cs4*half_rho*WEIGHTS[13]*(1.0f*cs2*(ux - uz)*((ux - uz) + two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[14] = inv_cs2_cs4*half_rho*WEIGHTS[14]*(1.0f*cs2*(ux + uz)*((ux + uz) - two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[15] = inv_cs2_cs4*half_rho*WEIGHTS[15]*(1.0f*cs2*(uy + uz)*((uy + uz) + two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[16] = inv_cs2_cs4*half_rho*WEIGHTS[16]*(1.0f*cs2*(uy - uz)*((uy - uz) - two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[17] = inv_cs2_cs4*half_rho*WEIGHTS[17]*(1.0f*cs2*(uy - uz)*((uy - uz) + two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[18] = inv_cs2_cs4*half_rho*WEIGHTS[18]*(1.0f*cs2*(uy + uz)*((uy + uz) - two_cs4) + two_cs2_cs4  - cs4*(ux2 + uy2 + uz2));
        f_eq[19] = inv_cs2_cs4*half_rho*WEIGHTS[19]*(1.0f*cs2*(ux + uy + uz)*((ux + uy + uz) + two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
        f_eq[20] = inv_cs2_cs4*half_rho*WEIGHTS[20]*(1.0f*cs2*(-ux + uy + uz)*((-ux + uy + uz) + two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
        f_eq[21] = inv_cs2_cs4*half_rho*WEIGHTS[21]*(1.0f*cs2*(ux - uy + uz)*((ux - uy + uz) + two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
        f_eq[22] = inv_cs2_cs4*half_rho*WEIGHTS[22]*(1.0f*cs2*(ux + uy - uz)*((ux + uy - uz) - two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
        f_eq[23] = inv_cs2_cs4*half_rho*WEIGHTS[23]*(1.0f*cs2*(ux + uy - uz)*((ux + uy - uz) + two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
        f_eq[24] = inv_cs2_cs4*half_rho*WEIGHTS[24]*(1.0f*cs2*(ux - uy + uz)*((ux - uy + uz) - two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
        f_eq[25] = inv_cs2_cs4*half_rho*WEIGHTS[25]*(1.0f*cs2*(-ux + uy + uz)*((-ux + uy + uz) - two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
        f_eq[26] = inv_cs2_cs4*half_rho*WEIGHTS[26]*(1.0f*cs2*(ux + uy + uz)*((ux + uy + uz) - two_cs4) + two_cs2_cs4 - cs4*(ux2 + uy2 + uz2));
    }
};

#endif // ! GUO_OUTLET_3D_H
