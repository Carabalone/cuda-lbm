#ifndef IBMUTILS_H
#define IBMUTILS_H

__host__ __device__
float delta(r) {
    return fabsf(r) <= 1 ? 
            1.0f - r 
            : 0
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

__host__ __device__ __forceinline__
void getAffectedNodeRange(float x, float y, int& x_start, int& y_start, int& x_end, int& y_end) {
    x_start = max(0, (int)floorf(x - 2.0f));
    y_start = max(0, (int)floorf(y - 2.0f));
    x_end = min(NX-1, (int)ceilf(x + 2.0f));
    y_end = min(NY-1, (int)ceilf(y + 2.0f));
}

// Determine if a lattice node is near a boundary point
__host__ __device__ __forceinline__
bool isNodeNearBoundary(float x_boundary, float y_boundary, int x_node, int y_node) {
    float dx = fabsf(x_boundary - x_node);
    float dy = fabsf(y_boundary - y_node);
    return (dx <= 2.0f && dy <= 2.0f);
}

#endif // ! IBMUTILS_H