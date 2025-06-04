#include "IBM/IBMUtils.cuh"
#include "IBM/IBMBody.cuh"

#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include "IBM/mesh/mesh_io.hpp"
#include "IBM/model_sampling/sample.hpp"
#include "IBM/model_sampling/sampler.hpp"
#include "IBM/model_sampling/utils.hpp"
#include "IBM/mesh/mesh.hpp"

namespace fs = std::filesystem;
namespace sm = sampler;

std::string create_csv_filename(const std::string& obj_filename, int target_samples) {
    fs::path obj_path(obj_filename);
    std::string stem = obj_path.stem().string();
    std::string parent_dir = obj_path.has_parent_path() ? obj_path.parent_path().string() : ".";
    
    std::stringstream csv_ss;
    csv_ss << parent_dir << "/" << stem;
    if (target_samples > 0) {
        csv_ss << "_target" << target_samples;
    }
    csv_ss << "_sampled_pts.csv";
    return csv_ss.str();
}

IBMBody load_from_obj(const std::string& filename, int target_samples /*=-1*/) {
    IBMBody body = {0, nullptr, nullptr};

    std::cout << "[INFO] Loading mesh using sampler::load_mesh_from_obj from: " << filename << std::endl;
    mesh::MeshData mesh = mesh::load_obj(filename);

    if (mesh.vertices.empty()) {
        std::cerr << "[ERROR] Failed to load mesh or mesh is empty using sampler::load_mesh_from_obj. Aborting." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<geom::Point3D> body_pts;
    bool perform_reduction = (target_samples > 0) && (target_samples < static_cast<int>(mesh.vertices.size()));

    if (perform_reduction) {
        std::cout << "[INFO] Target samples: " << target_samples 
                  << ". Loaded vertices: " << mesh.vertices.size() 
                  << ". Performing reduction." << std::endl;

        float volume = 0.0f;
        if (mesh.faces.empty()) {
            std::cout << "[ERROR] No faces loaded by sampler::load_mesh_from_obj" << std::endl;
            exit(EXIT_FAILURE);
        }
        volume = sampler::calculate_mesh_volume(mesh.vertices, mesh.faces);
        
        float r_max = sampler::calculate_r_max_3d(volume, target_samples);

        std::cout << std::fixed << std::setprecision(4);

        std::vector<sampler::Sample3> samples_to_reduce = sampler::Sample3::from_points(mesh.vertices);
        sampler::Sampler point_sampler(target_samples, r_max, samples_to_reduce);
        
        body_pts = point_sampler.eliminate_samples();

    } else {
        if (target_samples != -1 && target_samples != 0) {
             std::cout << "[INFO] Target samples (" << target_samples
                       << ") >= loaded points (" << mesh.vertices.size()
                       << "). No reduction needed." << std::endl;
        } else {
             std::cout << "[INFO] No sample reduction requested (target_samples=" << target_samples << ")." << std::endl;
        }
        body_pts = mesh.vertices;
    }

    if (body_pts.empty()){
        std::cerr << "[ERROR] No points to create IBMBody from file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    if (perform_reduction) {
        std::string output_csv_file = create_csv_filename(filename, (perform_reduction ? target_samples : -1));
        std::cout << "[INFO] Saving final points to: " << output_csv_file << std::endl;
        sampler::save_points_to_csv(body_pts, output_csv_file);
    }

    int num_pts = static_cast<int>(body_pts.size());
    body.num_points = num_pts;
    body.points = new float[3 * num_pts];     
    body.velocities = new float[3 * num_pts]; 

    for (int i = 0; i < num_pts; ++i) {
        body.points[3*i]   = body_pts[i].x();
        body.points[3*i+1] = body_pts[i].y();
        body.points[3*i+2] = body_pts[i].z();

        body.velocities[3*i]   = 0.0f;
        body.velocities[3*i+1] = 0.0f;
        body.velocities[3*i+2] = 0.0f;
    }

    std::cout << "[INFO] Created IBMBody with " << num_pts << " points from " << filename << "." << std::endl;
    return body;
}

IBMBody load_from_csv(const std::string& csv_filename) {
    IBMBody body = {0, nullptr, nullptr};
    std::ifstream file(csv_filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open CSV file " << csv_filename << std::endl;
        return body;
    }

    std::vector<geom::Point3D> loaded_points;
    std::string line_str;
    int line_num = 0;

    while (std::getline(file, line_str)) {
        line_num++;
        if (line_num == 1 && (line_str.find('x') != std::string::npos || line_str.find('X') != std::string::npos)) {
            // Skip header line if it looks like one
            std::cout << "[INFO] Skipping CSV header: " << line_str << std::endl;
            continue;
        }
        if (line_str.empty() || line_str[0] == '#') { // Skip empty lines or comments
            continue;
        }

        std::stringstream ss(line_str);
        std::string segment;
        geom::Point3D p;
        
        if (!std::getline(ss, segment, ',')) {
            std::cerr << "[WARNING] CSV line " << line_num << ": Could not read x value from: " << line_str << std::endl;
            continue;
        }
        try { p[0] = std::stof(segment); } catch (const std::exception& e) {
            std::cerr << "[WARNING] CSV line " << line_num << ": Invalid x value '" << segment << "': " << e.what() << std::endl; continue;
        }

        if (!std::getline(ss, segment, ',')) {
            std::cerr << "[WARNING] CSV line " << line_num << ": Could not read y value from: " << line_str << std::endl;
            continue;
        }
        try { p[1] = std::stof(segment); } catch (const std::exception& e) {
            std::cerr << "[WARNING] CSV line " << line_num << ": Invalid y value '" << segment << "': " << e.what() << std::endl; continue;
        }

        if (!std::getline(ss, segment, ',')) {
             try { p[2] = std::stof(segment); } catch (const std::exception& e) {
                std::cerr << "[WARNING] CSV line " << line_num << ": Invalid[2] value '" << segment << "': " << e.what() << std::endl; continue;
            }
        } else {
             try { p[2] = std::stof(segment); } catch (const std::exception& e) {
                std::cerr << "[WARNING] CSV line " << line_num << ": Invalid[2] value '" << segment << "': " << e.what() << std::endl; continue;
            }
        }
        loaded_points.push_back(p);
    }
    file.close();

    if (loaded_points.empty()) {
        std::cerr << "[WARNING] No valid points loaded from CSV file " << csv_filename << std::endl;
        return body;
    }

    int num_pts = static_cast<int>(loaded_points.size());
    body.num_points = num_pts;
    body.points = new float[3 * num_pts];     // Assuming 3D points
    body.velocities = new float[3 * num_pts]; // Assuming 3D points

    for (int i = 0; i < num_pts; ++i) {
        body.points[3 * i + 0] = loaded_points[i].x();
        body.points[3 * i + 1] = loaded_points[i].y();
        body.points[3 * i + 2] = loaded_points[i].z();
        body.velocities[3 * i + 0] = 0.0f;
        body.velocities[3 * i + 1] = 0.0f;
        body.velocities[3 * i + 2] = 0.0f;
    }

    std::cout << "[INFO] Loaded " << num_pts << " points from CSV file " << csv_filename << std::endl;
    return body;
}
