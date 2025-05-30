#ifndef IBMUTILS_H
#define IBMUTILS_H
#include "core/lbm_constants.cuh"
#include "IBM/IBMBody.cuh"
#include "IBM/model_sampling/sample.hpp"
#include <cfloat>
#include <cmath>

// assumes grid coordinates (system, but doesn't need to be aligned)
IBMBody create_cylinder(float cx, float cy, float r, int num_pts=16);
IBMBody create_sphere(float cx, float cy, float cz, float r, int n_theta, int n_phi);
IBMBody load_from_obj(const std::string& filename, int target_samples=-1); // 3d for now
IBMBody load_from_csv(const std::string& csv_filename);

struct AABB_3D {
    float min_x = FLT_MAX, min_y = FLT_MAX, min_z = FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX, max_z = -FLT_MAX;

    __host__ void update(float x, float y, float z = 0.0f) {
        min_x = fminf(min_x, x); max_x = fmaxf(max_x, x);
        min_y = fminf(min_y, y); max_y = fmaxf(max_y, y);
        min_z = fminf(min_z, z); max_z = fmaxf(max_z, z);
    }

    // TODO check z (2D case)
    __host__ bool is_valid() const {
        return min_x <= max_x && min_y <= max_y && min_z <= max_z;
    }
};

template <int dim>
__host__ void block_morton_sort(
    float* points, float* velocities, int num_ppts, float block_size, int morton_bits_per_dim = 10
);

__host__ __device__ __forceinline__
float delta(float r) {
    float absr = fabsf(r);
    return absr <= 1 ? 
            1.0f - absr :
            0;
}

__host__ __device__ __forceinline__
float kernel2D(float dx, float dy) {
    return delta(dx) * delta(dy);
}

__host__ __device__ __forceinline__
float kernel3D(float dx, float dy, float dz) {
    return delta(dx) * delta(dy) * delta(dz);
}

__host__ __device__ __forceinline__
float distance(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return sqrtf(dx*dx + dy*dy);
}

__host__ __device__ __forceinline__
float distance3D(float x1, float y1, float z1, float x2, float y2, float z2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    float dz = z1 - z2;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

#define IBM_SOA

#ifdef IBM_SOA
    __host__ __device__ __forceinline__
    int get_lag_vec_index(int node, int component, int num_pts) {
        return component * num_pts + node;
    }
#elif defined(IBM_CSOA)
    // TODO

    __host__ __device__ __forceinline__
    int get_lag_vec_index(int node, int component, int num_pts) {
        return node * dimensions + component;
    }
#else
    __host__ __device__ __forceinline__
    int get_lag_vec_index(int node, int component, int num_pts) {
        return node * dimensions + component;
    }
#endif // ! SOA

__host__ __device__ __forceinline__
int get_pt_index(int idx, int component) {
    return idx * dimensions + component;
}

#endif // ! IBMUTILS_H