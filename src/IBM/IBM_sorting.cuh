#ifndef IBM_SORTING_H
#define IBM_SORTING_H

#include "IBM/IBMUtils.cuh"
#include <vector>
#include <algorithm> 
#include <cmath>
#include <iostream>
#include <limits>
#include "IBM/mesh/mesh.hpp"

struct SortablePoint {
    float p[3];
    float v[3];
    int block_idx_1d;
    uint32_t morton_code;

    // for debug
    int original_idx;
};

static __host__ uint32_t calculate_morton_2d(uint32_t x, uint32_t y, int bits_per_dim) {
    uint32_t morton = 0;
    for (int i = 0; i < bits_per_dim; ++i) {
        morton |= ((x >> i) & 1) << (2 * i + 0);
        morton |= ((y >> i) & 1) << (2 * i + 1);
    }
    return morton;
}

static __host__ uint32_t calculate_morton_3d(uint32_t x, uint32_t y, uint32_t z, int bits_per_dim) {
    uint32_t morton = 0;
    for (int i = 0; i < bits_per_dim; ++i) {
        morton |= ((x >> i) & 1) << (3 * i + 0);
        morton |= ((y >> i) & 1) << (3 * i + 1);
        morton |= ((z >> i) & 1) << (3 * i + 2);
    }
    return morton;
}

template <int dim>
__host__ void block_morton_sort(
    float* points,
    float* velocities,
    int num_pts,
    int block_size,
    int morton_bits_per_dim) {

    if (num_pts == 0 || points == nullptr || velocities == nullptr) {
        return;
    }

    mesh::AABB body_aabb;
    for (int i = 0; i < num_pts; ++i) {
        float px = points[i * dim + 0];
        float py = points[i * dim + 1];
        float pz = (dim == 3) ? points[i * dim + 2] : 0.0f;
        body_aabb.update(px, py, pz);
    }

    if (!body_aabb.is_valid) {
        printf(
            "[WARNING] AABB for Morton sorting is invalid (num_points: %d). "
            "Skipping.\n",
            num_pts);
        return;
    }

    printf("AABB: min(%.3f, %.3f, %.3f), max(%.3f, %.3f, %.3f)\n",
           body_aabb.min_ext.x(), body_aabb.min_ext.y(), body_aabb.min_ext.z(),
           body_aabb.max_ext.x(), body_aabb.max_ext.y(), body_aabb.max_ext.z());

    float block_grid_origin_x = floorf(body_aabb.min_ext.x() / block_size) * block_size;
    float block_grid_origin_y = floorf(body_aabb.min_ext.y() / block_size) * block_size;
    float block_grid_origin_z = 0.0f;
    if constexpr (dim == 3) {
        block_grid_origin_z = floorf(body_aabb.min_ext.z() / block_size) * block_size;
    }

    float epsilon = static_cast<float>(block_size) * 1e-6f;
    int num_blocks_x = static_cast<int>(ceilf((body_aabb.max_ext.x() - block_grid_origin_x + epsilon) / block_size));
    int num_blocks_y = static_cast<int>(ceilf((body_aabb.max_ext.y() - block_grid_origin_y + epsilon) / block_size));
    int num_blocks_z = 1;
    if constexpr (dim == 3) {
        num_blocks_z = static_cast<int>(ceilf((body_aabb.max_ext.z() - block_grid_origin_z + epsilon) / block_size));
    }

    num_blocks_x = std::max(1, num_blocks_x);
    num_blocks_y = std::max(1, num_blocks_y);
    num_blocks_z = std::max(1, num_blocks_z);

    std::vector<SortablePoint> sortable_data(num_pts);
    uint32_t morton_coord_max_val = (1 << morton_bits_per_dim) - 1;

    for (int i = 0; i < num_pts; i++) {
        float px = points[i * dim + 0];
        float py = points[i * dim + 1];
        float pz = (dim == 3) ? points[i * dim + 2] : 0.0f;

        sortable_data[i].p[0] = px;
        sortable_data[i].p[1] = py;
        sortable_data[i].p[2] = (dim == 3) ? pz : 0.0f;

        sortable_data[i].v[0] = velocities[i * dim + 0];
        sortable_data[i].v[1] = velocities[i * dim + 1];
        if constexpr (dim == 3) {
            sortable_data[i].v[2] = velocities[i * dim + 2];
        } else {
            sortable_data[i].v[2] = 0.0f; // Ensure v[2] is initialized for 2D
        }
        sortable_data[i].original_idx = i;

        int bix = static_cast<int>(floorf((px - block_grid_origin_x) / block_size));
        int biy = static_cast<int>(floorf((py - block_grid_origin_y) / block_size));
        bix = std::max(0, std::min(bix, num_blocks_x - 1));
        biy = std::max(0, std::min(biy, num_blocks_y - 1));

        float local_x = px - (block_grid_origin_x + static_cast<float>(bix) * block_size);
        float local_y = py - (block_grid_origin_y + static_cast<float>(biy) * block_size);
        uint32_t mx = static_cast<uint32_t>(fmaxf(0.0f, fminf(local_x / block_size, 1.0f)) * morton_coord_max_val);
        uint32_t my = static_cast<uint32_t>(fmaxf(0.0f, fminf(local_y / block_size, 1.0f)) * morton_coord_max_val);

        if constexpr (dim == 2) {
            sortable_data[i].block_idx_1d = biy * num_blocks_x + bix;
            sortable_data[i].morton_code = calculate_morton_2d(mx, my, morton_bits_per_dim);
        } else if constexpr (dim == 3) {
            int biz = static_cast<int>(floorf((pz - block_grid_origin_z) / block_size));
            biz = std::max(0, std::min(biz, num_blocks_z - 1));
            sortable_data[i].block_idx_1d = biz * (num_blocks_x * num_blocks_y) + biy * num_blocks_x + bix;

            float local_z = pz - (block_grid_origin_z + static_cast<float>(biz) * block_size);
            uint32_t mz = static_cast<uint32_t>(fmaxf(0.0f, fminf(local_z / block_size, 1.0f)) * morton_coord_max_val);
            sortable_data[i].morton_code = calculate_morton_3d(mx, my, mz, morton_bits_per_dim);
        }
    }

    std::sort(sortable_data.begin(), sortable_data.end(),
        [](const SortablePoint& a, const SortablePoint& b) {
        if (a.block_idx_1d != b.block_idx_1d) {
            return a.block_idx_1d < b.block_idx_1d;
        }
        return a.morton_code < b.morton_code;
    });

    for (int i = 0; i < num_pts; i++) {
        #pragma unroll
        for (int comp = 0; comp < dim; comp++) {
            points[i * dim + comp] = sortable_data[i].p[comp];
            velocities[i * dim + comp] = sortable_data[i].v[comp];
        }
    }
    printf(
        "Morton sorting <dim=%d> complete for %d points. Block size L=%d. "
        "Morton bits=%d\n",
        dim, num_pts, block_size, morton_bits_per_dim);
        
}

#endif // ! IBM_SORTING_H
