#ifndef DEFAULT_INIT_H
#define DEFAULT_INIT_H

struct DefaultInit {

    __host__ __device__
    DefaultInit() { }

     __device__
    inline void operator()(float* rho, float* u, int node) {
        rho[node]       = 1.0f;
        u[2 * node]     = 0.0001f;
        u[2 * node + 1] = 0.0001f;
    }
};

#endif // ! DEFAULT_INIT_H
