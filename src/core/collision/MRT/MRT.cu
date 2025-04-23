// #include "MRT.cuh"

// template <>
// void MRT<2>::compute_forcing_term(float* F, float* u, float* force, int node) {

//     float fx = force[get_vec_index(node, 0)];
//     float fy = force[get_vec_index(node, 1)];
//     float ux = u[get_vec_index(node, 0)];
//     float uy = u[get_vec_index(node, 1)];

//     // Orthogonalized Guo Forcing scheme for MRT
//     // G. Silva, V. Semiao, J. Fluid Mech. 698, 282 (2012)
    
//     F[0] = 0.0f;                          // ρ (density) - conserved; no force contribution
//     F[1] = 6.0f * (fx*ux + fy*uy);        // e (energy)
//     F[2] = -6.0f * (fx*ux + fy*uy);       // ε (energy squared)
//     F[3] = fx;                            // jx (x-momentum)
//     F[4] = fy;                            // jy (y-momentum)
//     F[5] = -fx;                           // qx (x heat flux)
//     F[6] = -fy;                           // qy (y heat flux)
//     F[7] = 2.0f * (fx*ux - fy*uy);        // pxx (xx stress)
//     F[8] = fx*uy + fy*ux;                 // pxy (xy stress)
// }

// template <>
// void MRT<3>::compute_forcing_term(float* F, float* u, float* force, int node) {

//     float fx = force[get_vec_index(node, 0)];
//     float fy = force[get_vec_index(node, 1)];
//     float fz = force[get_vec_index(node, 2)];

//     float ux = u[get_vec_index(node, 0)];
//     float uy = u[get_vec_index(node, 1)];
//     float uz = u[get_vec_index(node, 2)];

//     // Sympy-generated cse
//     const float x1 = fz*uz;
//     const float x2 = fy*uy;
//     const float x4 = fx*ux;
//     const float x11 = 7.0f*fx;
//     const float x15 = fy*ux;
//     const float x27 = fy*uz;
//     const float x28 = fz*uy;

//     F[0] = 0.0f,
//     F[1] = -1.0f/24.0f*fx*uy + (17.0f/72.0f)*fx - 23.0f/72.0f*fy + (7.0f/24.0f)*fz*ux - 31.0f/72.0f*fz + (1.0f/24.0f)*uz*x11 - 1.0f/4.0f*x1 - 1.0f/24.0f*x15 - 11.0f/12.0f*x2 + (7.0f/24.0f)*x27 + (7.0f/24.0f)*x28 - 19.0f/12.0f*x4;
//     F[2] = (19.0f/24.0f)*fx*uy - 5.0f/24.0f*fy + (7.0f/24.0f)*fz*ux - 1.0f/24.0f*fz + (1.0f/24.0f)*uz*x11 + (7.0f/12.0f)*x1 + (1.0f/24.0f)*x11 + (19.0f/24.0f)*x15 - 11.0f/12.0f*x2 + (11.0f/24.0f)*x27 + (11.0f/24.0f)*x28 + (1.0f/4.0f)*x4;
//     F[3] = (1.0f/8.0f)*fx*uy - 5.0f/24.0f*fx*uz + (5.0f/72.0f)*fx - 3.0f/8.0f*fy - 5.0f/24.0f*fz*ux - 11.0f/72.0f*fz - 3.0f/4.0f*x1 + (1.0f/8.0f)*x15 + (5.0f/12.0f)*x2 + (11.0f/24.0f)*x27 + (11.0f/24.0f)*x28 + (5.0f/12.0f)*x4;
//     F[4] = -1.0f/8.0f*fx*uy - 1.0f/8.0f*fx*uz + (1.0f/24.0f)*fx - 1.0f/72.0f*fy - 1.0f/8.0f*fz*ux - 17.0f/72.0f*fz + (19.0f/12.0f)*x1 - 1.0f/8.0f*x15 + (11.0f/12.0f)*x2 + (1.0f/24.0f)*x27 + (1.0f/24.0f)*x28 + (3.0f/4.0f)*x4;
//     F[5] = -fx - 5.0f/18.0f*fy - 7.0f/18.0f*fz - 7.0f/3.0f*x1 - 8.0f/3.0f*x2 - 1.0f/6.0f*x27 - 1.0f/6.0f*x28 - 1.0f/2.0f*x4;
//     F[6] = -1.0f/3.0f*fx*uy + (4.0f/9.0f)*fx + (11.0f/18.0f)*fy - 1.0f/2.0f*fz - 2.0f/3.0f*x1 - 1.0f/3.0f*x15 + (1.0f/6.0f)*x27 + (1.0f/6.0f)*x28 + (1.0f/2.0f)*x4;
//     F[7] = -5.0f/24.0f*fx*uy - 1.0f/24.0f*fx*uz - 19.0f/72.0f*fx + (1.0f/72.0f)*fy - 1.0f/24.0f*fz*ux - 19.0f/72.0f*fz + (7.0f/12.0f)*x1 - 5.0f/24.0f*x15 - 1.0f/4.0f*x2 + (1.0f/8.0f)*x27 + (1.0f/8.0f)*x28 - 5.0f/12.0f*x4;
//     F[8] = -1.0f/24.0f*fx*uy + (11.0f/24.0f)*fx*uz + (1.0f/72.0f)*fx - 1.0f/24.0f*fy + (11.0f/24.0f)*fz*ux - 7.0f/72.0f*fz + (1.0f/12.0f)*x1 - 1.0f/24.0f*x15 + (1.0f/12.0f)*x2 - 1.0f/24.0f*x27 - 1.0f/24.0f*x28 - 1.0f/12.0f*x4;
//     F[9] = (1.0f/8.0f)*fx*uy + (1.0f/8.0f)*fx*uz + (5.0f/72.0f)*fx - 11.0f/72.0f*fy + (1.0f/8.0f)*fz*ux - 1.0f/24.0f*fz - 1.0f/12.0f*x1 + (1.0f/8.0f)*x15 + (5.0f/12.0f)*x2 - 5.0f/24.0f*x27 - 5.0f/24.0f*x28 - 1.0f/4.0f*x4;
//     F[10] = -1.0f/12.0f*fx*uy - 5.0f/12.0f*fx*uz - 31.0f/36.0f*fx - 5.0f/36.0f*fy - 5.0f/12.0f*fz*ux + (11.0f/36.0f)*fz + (5.0f/2.0f)*x1 - 1.0f/12.0f*x15 + (19.0f/6.0f)*x2 + (1.0f/12.0f)*x27 + (1.0f/12.0f)*x28 + (16.0f/3.0f)*x4;
//     F[11] = (1.0f/12.0f)*fx*uy - 5.0f/12.0f*fx*uz - 11.0f/12.0f*fx + (3.0f/4.0f)*fy - 5.0f/12.0f*fz*ux - 1.0f/12.0f*fz - 1.0f/3.0f*x1 + (1.0f/12.0f)*x15 + (19.0f/6.0f)*x2 - 1.0f/12.0f*x27 - 1.0f/12.0f*x28 - 5.0f/2.0f*x4;
//     F[12] = -1.0f/4.0f*fx*uy - 1.0f/36.0f*fx + (11.0f/12.0f)*fy + (7.0f/12.0f)*fz*ux + (31.0f/36.0f)*fz + (1.0f/12.0f)*uz*x11 + 3.0f*x1 - 1.0f/4.0f*x15 - 13.0f/6.0f*x2 + (5.0f/12.0f)*x27 + (5.0f/12.0f)*x28 - 2.0f/3.0f*x4;
//     F[13] = -1.0f/24.0f*fx*uy - 17.0f/24.0f*fx*uz + (65.0f/72.0f)*fx + (13.0f/72.0f)*fy - 17.0f/24.0f*fz*ux + (53.0f/72.0f)*fz - 13.0f/4.0f*x1 - 1.0f/24.0f*x15 - 23.0f/12.0f*x2 - 5.0f/24.0f*x27 - 5.0f/24.0f*x28 - 61.0f/12.0f*x4;
//     F[14] = -17.0f/24.0f*fx*uy - 17.0f/24.0f*fx*uz + (19.0f/24.0f)*fx - 25.0f/24.0f*fy - 17.0f/24.0f*fz*ux - 1.0f/24.0f*fz - 11.0f/12.0f*x1 - 17.0f/24.0f*x15 - 23.0f/12.0f*x2 - 13.0f/24.0f*x27 - 13.0f/24.0f*x28 + (13.0f/4.0f)*x4;
//     F[15] = -3.0f/8.0f*fx*uy + (19.0f/24.0f)*fx*uz - 17.0f/24.0f*fy + (19.0f/24.0f)*fz*ux - 71.0f/72.0f*fz - 9.0f/4.0f*x1 - 1.0f/72.0f*x11 - 3.0f/8.0f*x15 + (41.0f/12.0f)*x2 - 1.0f/24.0f*x27 - 1.0f/24.0f*x28 - 13.0f/12.0f*x4;
//     F[16] = (1.0f/8.0f)*fx*uy + (1.0f/8.0f)*fx*uz - 1.0f/24.0f*fx - 11.0f/72.0f*fy + (1.0f/8.0f)*fz*ux + (5.0f/72.0f)*fz - 7.0f/12.0f*x1 + (1.0f/8.0f)*x15 + (1.0f/12.0f)*x2 + (11.0f/24.0f)*x27 + (11.0f/24.0f)*x28 - 1.0f/4.0f*x4;
//     F[17] = -3.0f/8.0f*fx*uy - 3.0f/8.0f*fx*uz + (1.0f/8.0f)*fx - 1.0f/24.0f*fy - 3.0f/8.0f*fz*ux + (31.0f/24.0f)*fz - 29.0f/4.0f*x1 - 3.0f/8.0f*x15 - 13.0f/4.0f*x2 + (1.0f/8.0f)*x27 + (1.0f/8.0f)*x28 - 15.0f/4.0f*x4;
//     F[18] = fx - 5.0f/18.0f*fy + (5.0f/18.0f)*fz + (11.0f/3.0f)*x1 + (16.0f/3.0f)*x2 - 1.0f/6.0f*x27 - 1.0f/6.0f*x28 + (3.0f/2.0f)*x4;
//     F[19] = -1.0f/3.0f*fx*uy - 2.0f/9.0f*fx - 13.0f/18.0f*fy + (1.0f/6.0f)*fz + (4.0f/3.0f)*x1 - 1.0f/3.0f*x15 + (1.0f/6.0f)*x27 + (1.0f/6.0f)*x28 - 3.0f/2.0f*x4;
//     F[20] = -1.0f/24.0f*fx*uz + (17.0f/72.0f)*fx - 11.0f/72.0f*fy - 1.0f/24.0f*fz*ux + (29.0f/72.0f)*fz + (1.0f/24.0f)*uy*x11 - 11.0f/12.0f*x1 + (7.0f/24.0f)*x15 + (3.0f/4.0f)*x2 + (1.0f/8.0f)*x27 + (1.0f/8.0f)*x28 + (7.0f/12.0f)*x4;
//     F[21] = -1.0f/24.0f*fx*uy - 13.0f/24.0f*fx*uz + (1.0f/72.0f)*fx - 1.0f/24.0f*fy - 13.0f/24.0f*fz*ux - 7.0f/72.0f*fz + (1.0f/12.0f)*x1 - 1.0f/24.0f*x15 + (1.0f/12.0f)*x2 + (23.0f/24.0f)*x27 + (23.0f/24.0f)*x28 - 1.0f/12.0f*x4;
//     F[22] = -3.0f/8.0f*fx*uy + (1.0f/8.0f)*fx*uz + (13.0f/72.0f)*fy + (1.0f/8.0f)*fz*ux - 5.0f/24.0f*fz + (5.0f/12.0f)*x1 - 1.0f/72.0f*x11 - 3.0f/8.0f*x15 - 7.0f/12.0f*x2 + (7.0f/24.0f)*x27 + (7.0f/24.0f)*x28 + (1.0f/4.0f)*x4;
//     F[23] = -1.0f/3.0f*fx*uy - 1.0f/3.0f*fx*uz + (1.0f/9.0f)*fx - 1.0f/6.0f*fy - 1.0f/3.0f*fz*ux - 1.0f/18.0f*fz + (2.0f/3.0f)*x1 - 1.0f/3.0f*x15 - 1.0f/3.0f*x2 - 1.0f/6.0f*x27 - 1.0f/6.0f*x28 - 1.0f/2.0f*x4;
//     F[24] = -1.0f/2.0f*fx*uy + (1.0f/3.0f)*fx*uz - 1.0f/18.0f*fx - 1.0f/18.0f*fy + (1.0f/3.0f)*fz*ux + (4.0f/9.0f)*fz - 1.0f/2.0f*x1 - 1.0f/2.0f*x15 + (1.0f/3.0f)*x2 + (1.0f/3.0f)*x27 + (1.0f/3.0f)*x28 + (1.0f/3.0f)*x4;
//     F[25] = (1.0f/6.0f)*fx*uy - 1.0f/3.0f*fx*uz - 1.0f/6.0f*fx + (1.0f/9.0f)*fy - 1.0f/3.0f*fz*ux + (1.0f/18.0f)*fz - 1.0f/6.0f*x1 + (1.0f/6.0f)*x15 + (1.0f/3.0f)*x2 + (1.0f/6.0f)*x27 + (1.0f/6.0f)*x28 - 1.0f/6.0f*x4;
//     F[26] = -1.0f/24.0f*fx*uy - 5.0f/24.0f*fx*uz + (1.0f/72.0f)*fx - 1.0f/24.0f*fy - 5.0f/24.0f*fz*ux - 7.0f/72.0f*fz + (1.0f/12.0f)*x1 - 1.0f/24.0f*x15 + (1.0f/12.0f)*x2 + (7.0f/24.0f)*x27 + (7.0f/24.0f)*x28 - 1.0f/12.0f*x4;
// }