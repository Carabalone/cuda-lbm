#ifndef IBMUTILS_H
#define IBMUTILS_H
#include "core/lbm_constants.cuh"
#include "IBM/IBMBody.cuh"
#include "IBM/model_sampling/sample.hpp"
#include <cfloat>
#include <cmath>

IBMBody load_from_obj(const std::string& filename, int target_samples =-1);

constexpr int  R          = 2;     
constexpr int  KERNEL_W   = 4;   // nodes per dimension   (3)
constexpr int  OFFSET     = -1;

__host__ __device__ __forceinline__
float delta2(float r) {
    float absr = fabsf(r);
    return absr <= 1 ? 
            1.0f - absr :
            0;
}

__host__ __device__ __forceinline__
float delta4(float r) {
    float rabs = fabsf(r);
    
    // if (rabs < 2.0f) return 0.25f * (1 + cosf(M_PI * 0.5f * r));
    // else             return 0.0f;

    if (rabs < 1.0f)        return 0.125f * (3.0f - 2.0f * rabs + sqrtf(1 + 4.0f*rabs - 4.0f * r * r));
    else if (rabs  < 2.0f)  return 0.125f * (5.0f - 2.0f * rabs - sqrtf(-7.0f + 12.0f * rabs - 4.0f * r * r));
    else                    return 0.0f;
}

__host__ __device__ __forceinline__
float delta(float r) {
    return delta4(r);
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
