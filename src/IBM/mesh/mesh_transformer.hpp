#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cstdlib>

#include "IBM/mesh/mesh.hpp"
#include "IBM/mesh/mesh_io.hpp"
#include "IBM/math/mat4.hpp"
#include "IBM/IBMBody.cuh"
#include "IBM/model_sampling/sampler.hpp"
#include "IBM/IBM_generators.cuh"

namespace mesh {

class MeshTransformer {
private:
    MeshData mesh;
    geom::Point3D anchor;

    void update_anchor_to_centroid();

public:
    explicit MeshTransformer(const std::string& filename);
    explicit MeshTransformer(MeshData data);

    MeshTransformer& set_anchor_to_centroid();
    MeshTransformer& set_anchor(float x, float y, float z);

    MeshTransformer& translate(float tx, float ty, float tz);
    MeshTransformer& scale(float sx, float sy, float sz);
    MeshTransformer& scale_to_overall_size(float target_max_dimension);

    MeshTransformer& rotate_x(float angle_rad);
    MeshTransformer& rotate_y(float angle_rad);
    MeshTransformer& rotate_z(float angle_rad);

    MeshTransformer& move_anchor_to_world(float wx, float wy, float wz);
    MeshTransformer& move_centroid_to_world(float wx, float wy, float wz);

    MeshData collect_mesh() const;
    std::vector<geom::Point3D> collect_points() const;
    AABB get_aabb() const;
    geom::Point3D get_anchor() const;

    void collect_file(const std::string& filepath) const;

    bool fits_in_domain(
        float domain_max_x, float domain_max_y, float domain_max_z,
        float domain_min_x = 0.0f, float domain_min_y = 0.0f, float domain_min_z = 0.0f) const;

    IBMBody collect_ibm_body() const;
    IBMBody collect_ibm_body(int target_num_samples) const;
};

} // namespace mesh
