#include "IBM/mesh/mesh_transformer.hpp"
#include <algorithm>
#include <iostream>

namespace mesh {

void MeshTransformer::update_anchor_to_centroid() {
    AABB current_box = get_aabb();
    if (!current_box.is_valid) {
        std::cerr << "[ERROR] MeshTransformer: Cannot initialize/update anchor to centroid, mesh AABB is invalid." << std::endl;
        exit(EXIT_FAILURE);
    }
    anchor = current_box.get_center();
}

MeshTransformer::MeshTransformer(const std::string& obj_filepath) {
    mesh = mesh::load_obj(obj_filepath);
    if (mesh.vertices.empty()) {
        std::cerr << "[ERROR] MeshTransformer: Failed to load mesh or mesh is empty from \"" + obj_filepath + "\". Cannot proceed." << std::endl;
        exit(EXIT_FAILURE);
    }
    update_anchor_to_centroid();
}

MeshTransformer::MeshTransformer(MeshData initial_mesh_data)
    : mesh(std::move(initial_mesh_data)) {
    if (mesh.vertices.empty()) {
        std::cerr << "[ERROR] MeshTransformer: Provided MeshData is empty. Cannot proceed." << std::endl;
        exit(EXIT_FAILURE);
    }
    update_anchor_to_centroid();
}

MeshTransformer& MeshTransformer::set_anchor_to_centroid() {
    update_anchor_to_centroid();
    return *this;
}

MeshTransformer& MeshTransformer::set_anchor(float x, float y, float z) {
    anchor = {x, y, z};
    return *this;
}

MeshTransformer& MeshTransformer::translate(float tx, float ty, float tz) {
    math::Mat4 t_mat = math::Mat4::translation(tx, ty, tz);
    for (auto& v : mesh.vertices) {
        v = t_mat.transform_point(v);
    }
    anchor = t_mat.transform_point(anchor);
    return *this;
}

MeshTransformer& MeshTransformer::scale(float sx, float sy, float sz) {

    math::Mat4 to_anchor_origin = math::Mat4::translation(-anchor.x(), -anchor.y(), -anchor.z());
    math::Mat4 scale_mat = math::Mat4::scaling(sx, sy, sz);
    math::Mat4 from_anchor_origin = math::Mat4::translation(anchor.x(), anchor.y(), anchor.z());
    math::Mat4 final_transform = from_anchor_origin * scale_mat * to_anchor_origin;

    for (auto& v : mesh.vertices) {
        v = final_transform.transform_point(v);
    }
    return *this;
}

MeshTransformer& MeshTransformer::scale_to_overall_size(float target_max_dimension) {

    AABB current_box = get_aabb();

    geom::Point3D current_size = current_box.get_size();
    float max_current_dim = std::max(
            {std::abs(current_size.x()), std::abs(current_size.y()), std::abs(current_size.z())}
    );

    float s_factor = target_max_dimension / max_current_dim;
    return scale(s_factor, s_factor, s_factor);
}

MeshTransformer& MeshTransformer::rotate_x(float angle_rad) {

    math::Mat4 to_anchor_origin = math::Mat4::translation(-anchor.x(), -anchor.y(), -anchor.z());
    math::Mat4 rot_m = math::Mat4::rotationX(angle_rad);
    math::Mat4 from_anchor_origin = math::Mat4::translation(anchor.x(), anchor.y(), anchor.z());
    math::Mat4 final_transform = from_anchor_origin * rot_m * to_anchor_origin;

    for (auto& v : mesh.vertices) {
        v = final_transform.transform_point(v);
    }
    return *this;
}

MeshTransformer& MeshTransformer::rotate_y(float angle_rad) {
    
    math::Mat4 to_anchor_origin = math::Mat4::translation(-anchor.x(), -anchor.y(), -anchor.z());
    math::Mat4 rot_m = math::Mat4::rotationY(angle_rad);
    math::Mat4 from_anchor_origin = math::Mat4::translation(anchor.x(), anchor.y(), anchor.z());
    math::Mat4 final_transform = from_anchor_origin * rot_m * to_anchor_origin;

    for (auto& v : mesh.vertices) {
        v = final_transform.transform_point(v);
    }
    return *this;
}

MeshTransformer& MeshTransformer::rotate_z(float angle_rad) {
    
    math::Mat4 to_anchor_origin = math::Mat4::translation(-anchor.x(), -anchor.y(), -anchor.z());
    math::Mat4 rot_m = math::Mat4::rotationZ(angle_rad);
    math::Mat4 from_anchor_origin = math::Mat4::translation(anchor.x(), anchor.y(), anchor.z());
    math::Mat4 final_transform = from_anchor_origin * rot_m * to_anchor_origin; // Corrected: was from_origin

    for (auto& v : mesh.vertices) {
        v = final_transform.transform_point(v);
    }
    return *this;
}

MeshTransformer& MeshTransformer::move_anchor_to_world(float wx, float wy, float wz) {

    float tx = wx - anchor.x();
    float ty = wy - anchor.y();
    float tz = wz - anchor.z();
    
    return translate(tx, ty, tz);
}

MeshTransformer& MeshTransformer::move_centroid_to_world(float wx, float wy, float wz) {

    AABB current_box = get_aabb();
    geom::Point3D current_centroid = current_box.get_center();
    float tx = wx - current_centroid.x();
    float ty = wy - current_centroid.y();
    float tz = wz - current_centroid.z();
    
    return translate(tx, ty, tz);
}

MeshData MeshTransformer::collect_mesh() const {
    return mesh;
}

std::vector<geom::Point3D> MeshTransformer::collect_points() const {
    return mesh.vertices;
}

AABB MeshTransformer::get_aabb() const {
    return mesh::AABB(mesh.vertices);
}

geom::Point3D MeshTransformer::get_anchor() const {
    return anchor;
}

void MeshTransformer::collect_file(const std::string& filepath) const {
    mesh::save_obj(mesh, filepath);
}

bool MeshTransformer::fits_in_domain(

    float domain_max_x, float domain_max_y, float domain_max_z,
    float domain_min_x, float domain_min_y, float domain_min_z) const {
    AABB current_box = get_aabb();
    
    bool fits = current_box.min_ext.x() >= domain_min_x && current_box.max_ext.x() <= domain_max_x &&
                current_box.min_ext.y() >= domain_min_y && current_box.max_ext.y() <= domain_max_y &&
                current_box.min_ext.z() >= domain_min_z && current_box.max_ext.z() <= domain_max_z;

    if (!fits) {
        std::cout << "[INFO] Transformed mesh AABB: "
                  << "min(" << current_box.min_ext.x() << "," << current_box.min_ext.y() << "," << current_box.min_ext.z() << "), "
                  << "max(" << current_box.max_ext.x() << "," << current_box.max_ext.y() << "," << current_box.max_ext.z() << ") "
                  << "does not fit domain ["
                  << domain_min_x << "-" << domain_max_x << ", "
                  << domain_min_y << "-" << domain_max_y << ", "
                  << domain_min_z << "-" << domain_max_z << "]." << std::endl;
    } else {
         std::cout << "[INFO] Transformed mesh AABB fits within the domain." << std::endl;
    }
    return fits;
}

IBMBody MeshTransformer::collect_ibm_body() const {
    if (mesh.vertices.empty()) {
        std::cerr << "[Warning] MeshTransformer::collect_ibm_body_direct: No vertices." << std::endl;
        return {0, nullptr, nullptr};
    }
    return body_from_points<3>(mesh.vertices);
}

IBMBody MeshTransformer::collect_ibm_body(int target_num_samples) const {
    if (mesh.vertices.empty()) {
        std::cerr << "[Warning] MeshTransformer::collect_ibm_body_with_sampling: No vertices." << std::endl;
        return {0, nullptr, nullptr};
    }

    const std::vector<geom::Point3D>& vertices_to_sample = mesh.vertices;

    float mesh_volume = 0.0f;
    
    mesh_volume = sampler::calculate_mesh_volume(vertices_to_sample, mesh.faces);

    float r_max = sampler::calculate_r_max_3d(mesh_volume, target_num_samples);

    std::vector<sampler::Sample3> samples_to_reduce = sampler::Sample3::from_points(vertices_to_sample);
    std::vector<geom::Point3D> sampled_points;

    if (!samples_to_reduce.empty()) {
        sampler::Sampler point_sampler(target_num_samples, r_max, samples_to_reduce);
        sampled_points = point_sampler.eliminate_samples();
    }

    return body_from_points<3>(sampled_points);
}

} // namespace mesh
