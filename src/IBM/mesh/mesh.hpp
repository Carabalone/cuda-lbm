#pragma once

#include <vector>
#include <string>
#include <cfloat>    
#include <algorithm> 
#include <iostream>  

#include "IBM/geometry/point.hpp"

namespace mesh {

struct Face {
    int v_indices[3];
};

struct MeshData {
    std::vector<geom::Point3D> vertices;
    std::vector<Face> faces;
};

struct AABB {
    geom::Point3D min_ext;
    geom::Point3D max_ext;
    bool is_valid;

    AABB()
        : min_ext({FLT_MAX, FLT_MAX, FLT_MAX}),
          max_ext({-FLT_MAX, -FLT_MAX, -FLT_MAX}),
          is_valid(false) {}

    explicit AABB(const std::vector<geom::Point3D>& points)
        : min_ext({FLT_MAX, FLT_MAX, FLT_MAX}),
          max_ext({-FLT_MAX, -FLT_MAX, -FLT_MAX}),
          is_valid(false) {
        if (points.empty()) {
            return;
        }
        for (const auto& p : points) {
            update(p);
        }
    }

    explicit AABB(const MeshData& mesh_data)
        : min_ext({FLT_MAX, FLT_MAX, FLT_MAX}),
          max_ext({-FLT_MAX, -FLT_MAX, -FLT_MAX}),
          is_valid(false) {

        if (mesh_data.vertices.empty()) {
            return;
        }

        for (const auto& v : mesh_data.vertices) {
            update(v);
        }
    }


    void update(const geom::Point3D& p) {
        min_ext[0] = std::min(min_ext[0], p.x());
        max_ext[0] = std::max(max_ext[0], p.x());
        min_ext[1] = std::min(min_ext[1], p.y());
        max_ext[1] = std::max(max_ext[1], p.y());
        min_ext[2] = std::min(min_ext[2], p.z());
        max_ext[2] = std::max(max_ext[2], p.z());
        is_valid = true;
    }

    void update(float px, float py, float pz = 0.0f) {
        min_ext[0] = std::min(min_ext[0], px);
        max_ext[0] = std::max(max_ext[0], px);
        min_ext[1] = std::min(min_ext[1], py);
        max_ext[1] = std::max(max_ext[1], py);
        min_ext[2] = std::min(min_ext[2], pz);
        max_ext[2] = std::max(max_ext[2], pz);
        is_valid = true;
    }

    geom::Point3D get_center() const {

        return {(min_ext[0] + max_ext[0]) * 0.5f,
                (min_ext[1] + max_ext[1]) * 0.5f,
                (min_ext[2] + max_ext[2]) * 0.5f};
    }

    geom::Point3D get_size() const {

        return {max_ext[0] - min_ext[0],
                max_ext[1] - min_ext[1],
                max_ext[2] - min_ext[2]};
    }

    void reset() {
        min_ext = {FLT_MAX, FLT_MAX, FLT_MAX};
        max_ext = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
        is_valid = false;
    }
};

} // namespace mesh

