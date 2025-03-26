#ifndef IBMUTILS_H
#define IBMUTILS_H
#include "core/lbm_constants.cuh"

__host__ __device__
float delta(float r) {
    return fabsf(r) <= 1 ? 
            1.0f - r 
            : 0;
}

__host__ __device__ __forceinline__
float kernel2D(float dx, float dy) {
    return delta(dx) * delta(dy);
}

__host__ __device__ __forceinline__
float distance(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return sqrtf(dx*dx + dy*dy);
}
#endif // ! IBMUTILS_H