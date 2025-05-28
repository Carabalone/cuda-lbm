#pragma once
#include "points.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace sampler {


    std::vector<Point3> generate_sphere_points(
        int num_points,
        float sphere_radius) {
        std::vector<Point3> points_on_sphere;
        points_on_sphere.reserve(num_points);

        if (num_points <= 0) {
            return points_on_sphere;
        }
        if (num_points == 1) {
            points_on_sphere.push_back({0.0f, sphere_radius, 0.0f});
            return points_on_sphere;
        }

        const float golden_angle_increment = static_cast<float>(M_PI) * (3.0f - std::sqrt(5.0f)); // Approx 2.3999 rad

        for (int i = 0; i < num_points; ++i) {
            float y = 1.0f - (static_cast<float>(i) / (num_points - 1)) * 2.0f;
            float radius_at_y = std::sqrt(1.0f - y * y);
            float theta = static_cast<float>(i) * golden_angle_increment;

            float x = std::cos(theta) * radius_at_y;
            float z = std::sin(theta) * radius_at_y;

            points_on_sphere.push_back({
                x * sphere_radius,
                y * sphere_radius,
                z * sphere_radius
            });
        }
        return points_on_sphere;
    }

    MeshData load_mesh_from_obj(const std::string& filename) {
        MeshData mesh_data;
        std::ifstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return mesh_data;
        }

        std::string line_str;
        while (std::getline(file, line_str)) {
            std::stringstream ss(line_str);
            std::string keyword;
            ss >> keyword;

            if (keyword == "v") {
                Point3 p;
                if (ss >> p.x >> p.y >> p.z) {
                    mesh_data.vertices.push_back(p);
                } else {
                    std::cerr << "[WARNING] Could not parse vertex line: "
                            << line_str << std::endl;
                }
            } else if (keyword == "f") {
                Face f;
                bool success = true;
                int vertex_count_on_face = 0;
                for (int i = 0; i < 3; ++i) {
                    std::string face_val_str;
                    if (!(ss >> face_val_str)) {
                        if (i < 3 && vertex_count_on_face < 3) success = false; // Not enough vertices for a triangle
                        break; 
                    }
                    vertex_count_on_face++;
                    size_t first_slash = face_val_str.find('/');
                    std::string v_idx_str = face_val_str.substr(0, first_slash);
                    try {
                        // obj is 1-index based.
                        f.v_indices[i] = std::stoi(v_idx_str) - 1;
                    } catch (const std::invalid_argument& ia) {
                        std::cerr << "[WARNING] Invalid face vertex index: "
                                << v_idx_str << " from line: " << line_str
                                << std::endl;
                        success = false;
                        break;
                    } catch (const std::out_of_range& oor) {
                        std::cerr << "[WARNING] Face vertex index out of range: "
                                << v_idx_str << " from line: " << line_str
                                << std::endl;
                        success = false;
                        break;
                    }
                }
                if (success && vertex_count_on_face >= 3) {
                    mesh_data.faces.push_back(f);
                } else if (vertex_count_on_face > 0 && vertex_count_on_face < 3) {
                    std::cerr << "[WARNING] Face line has < 3 vertices: "
                            << line_str << std::endl;
                }
            }
        }
        file.close();
        if (!mesh_data.vertices.empty()) {
            std::cout << "[INFO] Loaded " << mesh_data.vertices.size()
                    << " vertices and " << mesh_data.faces.size()
                    << " faces from " << filename << std::endl;
        }
        return mesh_data;
    }

    std::vector<Point3> load_points_from_obj(const std::string& filename) {
        std::vector<Point3> vertices;
        std::ifstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return vertices;
        }

        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string keyword;
            ss >> keyword;

            if (keyword == "v") {
                Point3 p;
                if (ss >> p.x >> p.y >> p.z) {
                    vertices.push_back(p);
                } else {
                    std::cerr << "[WARNING] Could not parse vertex line: " << line
                            << std::endl;
                }
            }
        }

        file.close();
        return vertices;
    }

    void save_points_to_csv(const std::vector<Point3>& points, const std::string& filename) {
        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return;
        }
        outfile << "x,y,z\n"; // CSV Header
        for (const auto& p : points) {
            outfile << p.x << "," << p.y << "," << p.z << "\n";
        }
        outfile.close();
        std::cout << "[INFO] Saved " << points.size() << " points to " << filename << std::endl;
    }

}