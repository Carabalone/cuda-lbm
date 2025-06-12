#ifndef REGULARIZED_BOUNCE_BACK_H
#define REGULARIZED_BOUNCE_BACK_H

//TODO: Reduce reused code from the two conditions here and in regularized inlet

struct RegularizedBounceBack {
    bool x, y;
    
    __host__ __device__
    RegularizedBounceBack(bool x=true, bool y=true) : x(x), y(y) {}
    
    __device__
    inline void apply(float* f, float* f_back, float ux_node, float uy_node, int node) {
        const int node_x = node % NX;
        const int node_y = node / NX;
        
        bool is_unknown[quadratures] = {false};
        for (int i = 1; i < quadratures; i++) {

            if (node_x == 0) {
                is_unknown[i] = (C[2*i] > 0);
            }
            else if (node_x == NX-1) {
                is_unknown[i] = (C[2*i] < 0);
            }
            else if (node_y == 0) {
                is_unknown[i] = (C[2*i+1] > 0);
            }
            else if (node_y == NY-1) {
                is_unknown[i] = (C[2*i+1] < 0);
            }

        }
        
        float rho = 0.0f;
        // if(node_x == 100)
        //     printf("\nnode at y=%d known are: ", node_y);
        for (int i = 0; i < quadratures; i++) {
            if (!is_unknown[i]) {
                int w = is_unknown[OPP[i]] ? 2 : 1;
                rho += w * f[get_node_index(node, i)]; // doesnt need to divide by 1 - or 1 + uy or ux because u = (0,0)
                // if(node_x == 100)
                    // printf("(node: %d, w: %d)", i, w);
            }
        }

        const float ux = 0.0f;
        const float uy = 0.0f;
        float cs2 = 1.0f / 3.0f;
        
        float f_eq[quadratures];
        for (int q = 0; q < quadratures; q++) {
            float c_dot_u = C[2*q] * ux + C[2*q+1] * uy;
            float u_dot_u = ux*ux + uy*uy;  // Zero for no-slip
            f_eq[q] = WEIGHTS[q] * rho * (1.0f + c_dot_u/cs2 + 
                      c_dot_u*c_dot_u/(2.0f*cs2*cs2) - u_dot_u/(2.0f*cs2));
        }

        // bounce-back of non-equilibrium directions
        for (int q = 0; q < quadratures; q++) {
            if (is_unknown[q]) {
                int opp = OPP[q];
                f[get_node_index(node, q)] = f_eq[q] + (f[get_node_index(node, opp)] - f_eq[opp]);
            }
        }
        
        // Π(1) = ∑ᵢ fᵢ - cs² ρ I - ρ uu
        float Pi_xx = 0.0f, Pi_yy = 0.0f, Pi_xy = 0.0f;
        
        for (int q = 0; q < quadratures; q++) {
            float cx = C[2*q], cy = C[2*q+1];
            int idx = get_node_index(node, q);
            Pi_xx += cx * cx * f[idx];
            Pi_yy += cy * cy * f[idx];
            Pi_xy += cx * cy * f[idx];
        }
        
        Pi_xx -= cs2 * rho + rho * ux * ux;
        Pi_yy -= cs2 * rho + rho * uy * uy;
        Pi_xy -= rho * ux * uy;
        
        // regularize all populations
        float new_rho = 0.0f;
        for (int q = 0; q < quadratures; q++) {
            float cx = C[2*q], cy = C[2*q+1];
            float Q_xx = cx*cx - cs2;
            float Q_yy = cy*cy - cs2;
            float Q_xy = cx*cy;
            
            float f_neq = (WEIGHTS[q]/(2.0f*cs2*cs2)) * 
                        (Q_xx*Pi_xx + Q_yy*Pi_yy + 2.0f*Q_xy*Pi_xy);
            
            int idx = get_node_index(node, q);
            f[idx] = f_eq[q] + f_neq;
            new_rho += f[idx];
        }
        // if (new_rho > 1.2f || new_rho < 0.9f)
        //     printf("new_rho at node(%d, %d): %.4f\n", (node % NX), (node / NX), new_rho);
    }
};

struct RegularizedCornerBounceBack {
    __host__ __device__
    RegularizedCornerBounceBack() {}
    
    __device__ static
    inline void apply(float* f, float* f_back, float ux_node, float uy_node, int node) {
        int idx = get_node_index(node);
        const int node_x = node % NX;
        const int node_y = node / NX;
        
        bool is_left = (node_x == 0);
        bool is_right = (node_x == NX-1);
        bool is_bottom = (node_y == 0);
        bool is_top = (node_y == NY-1);
        
        bool is_corner = (is_left || is_right) && (is_bottom || is_top);
        
        if (!is_corner) {
            return;
        }
        
        bool is_unknown[quadratures] = {false};
        for (int i = 1; i < quadratures; i++) {
            int cx = C[2*i];
            int cy = C[2*i+1];
            
            is_unknown[i] = (is_left && cx > 0) || 
                            (is_right && cx < 0) || 
                            (is_bottom && cy > 0) || 
                            (is_top && cy < 0);
        }

        // 3. Compute macroscopic values (density/temperature)

        int diag_x = is_left ? node_x + 1 : node_x - 1;
        int diag_y = is_bottom ? node_y + 1 : node_y - 1;
        
        diag_x = max(1, min(diag_x, NX-2));
        diag_y = max(1, min(diag_y, NY-2));
        
        int diag_node = diag_y * NX + diag_x;

        float rho = 0.0f;
        for (int i = 0; i < quadratures; i++) {
            rho += f[get_node_index(diag_node, i)];
        }

        // printf("Corner node (%d,%d) using density %.4f from diagonal (%d,%d)\n", 
        //        node_x, node_y, rho, diag_x, diag_y);

        
        const float ux = 0.0f;
        const float uy = 0.0f;
        
        for (int i = 0; i < quadratures; i++) {
            if (is_unknown[i]) {
                float f_eq_i = compute_equilibrium(i, rho, ux, uy);
                
                int opp_i = OPP[i];
                
                float f_eq_opp = compute_equilibrium(opp_i, rho, ux, uy);
                
                f[get_node_index(node, i)] = f_eq_i - (f[get_node_index(node, opp_i)] - f_eq_opp);
            }
        }
        
        regularize_distributions(f, node, rho, ux, uy);


        // printf("LEFT NODE(%d,%d): Wall=[x:%d,y:%d] | rho=%.5f | u=(%.5f,%.5f) | KNOWN:[%s%s%s%s%s%s%s%s%s] | UNKNOWN:[%s%s%s%s%s%s%s%s%s]\n" /* | f_eq[1]=%.5f | f_eq[3]=%.5f | Pi=(%.5f,%.5f,%.5f)\n"*/, 
        //     node_x, node_y, node_x, node_y, rho, ux, uy, 
        //     !is_unknown[0] ? "0," : "", !is_unknown[1] ? "1," : "", !is_unknown[2] ? "2," : "", 
        //     !is_unknown[3] ? "3," : "", !is_unknown[4] ? "4," : "", !is_unknown[5] ? "5," : "", 
        //     !is_unknown[6] ? "6," : "", !is_unknown[7] ? "7," : "", !is_unknown[8] ? "8," : "",
        //     is_unknown[0] ? "0," : "", is_unknown[1] ? "1," : "", is_unknown[2] ? "2," : "", 
        //     is_unknown[3] ? "3," : "", is_unknown[4] ? "4," : "", is_unknown[5] ? "5," : "", 
        //     is_unknown[6] ? "6," : "", is_unknown[7] ? "7," : "", is_unknown[8] ? "8," : ""/*,
        //     f_eq[1], f_eq[3], Pi_xx, Pi_yy, Pi_xy*/);
    }
    
    __device__ static
    inline float compute_equilibrium(int i, float rho, float ux, float uy) {
        float cs2 = 1.0f / 3.0f;
        float c_dot_u = C[2*i] * ux + C[2*i+1] * uy;
        float u_dot_u = ux*ux + uy*uy;
        
        return WEIGHTS[i] * rho * (1.0f + c_dot_u/cs2 + 
                c_dot_u*c_dot_u/(2.0f*cs2*cs2) - u_dot_u/(2.0f*cs2));
    }
    
    __device__ static
    inline void regularize_distributions(float* f, int node, float rho, float ux, float uy) {
        float cs2 = 1.0f / 3.0f;
        
        float Pi_xx = 0.0f, Pi_yy = 0.0f, Pi_xy = 0.0f;
        
        for (int q = 0; q < quadratures; q++) {
            float f_eq = compute_equilibrium(q, rho, ux, uy);
            float f_neq = f[get_node_index(node, q)] - f_eq;
            
            float cx = C[2*q], cy = C[2*q+1];
            Pi_xx += cx * cx * f_neq;
            Pi_yy += cy * cy * f_neq;
            Pi_xy += cx * cy * f_neq;
        }
        
        for (int q = 0; q < quadratures; q++) {
            float f_eq = compute_equilibrium(q, rho, ux, uy);
            
            float cx = C[2*q], cy = C[2*q+1];
            float Q_xx = cx*cx - cs2;
            float Q_yy = cy*cy - cs2;
            float Q_xy = cx*cy;
            
            float f_neq = (WEIGHTS[q]/(2.0f*cs2*cs2)) * 
                        (Q_xx*Pi_xx + Q_yy*Pi_yy + 2.0f*Q_xy*Pi_xy);
            
            f[get_node_index(node, q)] = f_eq + f_neq;
        }
    }
};

#endif // ! REGULARIZED_BOUNCE_BACK_H
