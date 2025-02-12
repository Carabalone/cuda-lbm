#ifndef DEFAULT_INIT_H
#define DEFAULT_INIT_H

struct DefaultInit {

    __host__ __device__
    static inline void apply(float* rho, float* u, int node) {
        rho[node]       = 1.0f;
        u[2 * node]     = 0.00f;
        u[2 * node + 1] = 0.05f;
    }
};

#endif // ! DEFAULT_INIT_H
