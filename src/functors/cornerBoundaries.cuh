#ifndef CORNER_H
#define CORNER_H

struct CornerBoundary {
    __device__ static void apply_top_left(float* f, float* f_back, const int* C, const int* OPP, int node) {
        node = get_node_index(node);
        
        const float ux = 0.03f;
        const float uy = 0.0f;
        
        float rho = (f[0 + node] + f[3 + node] + f[4 + node] + 
                    f[6 + node] + f[7 + node]) / (1.0f - ux - uy);
        
        // f1 (right) - from inlet condition
        f[1 + node] = f[3 + node] + (2.0f/3.0f) * rho * ux;
        
        // f2 (up) - from wall condition
        f[2 + node] = f[4 + node];
        
        // f5 (up-right) - combined effect
        f[5 + node] = f[7 + node] + (1.0f/6.0f) * rho * (ux + uy);
        
        // f8 (down-right) - from inlet
        f[8 + node] = f[6 + node] + (1.0f/6.0f) * rho * (ux - uy);
    }
    
    __device__ static void apply_bottom_left(float* f, float* f_back, const int* C, const int* OPP, int node) {
        node = get_node_index(node);
        
        const float ux = 0.03f;
        const float uy = 0.0f;
        
        float rho = (f[0 + node] + f[2 + node] + f[3 + node] + 
                    f[5 + node] + f[6 + node]) / (1.0f - ux + uy);
        
        // f1 (right) - from inlet condition
        f[1 + node] = f[3 + node] + (2.0f/3.0f) * rho * ux;
        
        // f4 (down) - from wall condition
        f[4 + node] = f[2 + node];
        
        // f5 (up-right) - from inlet
        f[5 + node] = f[7 + node] + (1.0f/6.0f) * rho * (ux + uy);
        
        // f8 (down-right) - combined effect
        f[8 + node] = f[6 + node] + (1.0f/6.0f) * rho * (ux - uy);
    }
};

#endif // ! CORNER_H