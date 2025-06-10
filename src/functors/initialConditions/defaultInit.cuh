#ifndef DEFAULT_INIT_H
#define DEFAULT_INIT_H

template <int dim>
struct DefaultInit {

    __host__ __device__
    DefaultInit() { }

    __device__
    inline void apply_forces(float* rho, float* u, float* force, int node) {

        #pragma unroll
        for (int d = 0; d < dim; ++d) {
            force[get_vec_index(node, d)] = 0.0f;
        }

    }

     __device__
    inline void operator()(float* rho, float* u, float* force, int node) {
        rho[node] = 1.0f;

        #pragma unroll
        for (int d = 0; d < dim; ++d) {
            u[get_vec_index(node, d)] = 0.0f;
        }

        apply_forces(rho, u, force, node);
    }
};

#endif // ! DEFAULT_INIT_H
