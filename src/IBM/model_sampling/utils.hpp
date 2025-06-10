#ifndef SAMPLER_UTILS_H
#define SAMPLER_UTILS_H

#pragma once
#include "IBM/geometry/point.hpp"
#include <filesystem>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fs = std::filesystem;

namespace sampler {


    inline std::vector<geom::Point3D> generate_sphere_points(
        int num_points,
        float sphere_radius) {
        std::vector<geom::Point3D> points_on_sphere;
        points_on_sphere.reserve(num_points);

        if (num_points <= 0) {
            return points_on_sphere;
        }
        if (num_points == 1) {
            points_on_sphere.push_back({0.0f, sphere_radius, 0.0f});
            return points_on_sphere;
        }

        const float golden_angle_increment = static_cast<float>(M_PI) * (3.0f - std::sqrt(5.0f));

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

    inline std::vector<geom::Point3D> load_points_from_obj(const std::string& filename) {
        std::vector<geom::Point3D> vertices;
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
                geom::Point3D p;
                if (ss >> p[0] >> p[1] >> p[2]) {
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

    inline void save_points_to_csv(const std::vector<geom::Point3D>& points, const std::string& filename) {
        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return;
        }
        outfile << "x,y,z\n"; // CSV Header
        for (const auto& p : points) {
            outfile << p[0] << "," << p[1] << "," << p[2] << "\n";
        }
        outfile.close();
        std::cout << "[INFO] Saved " << points.size() << " points to " << filename << std::endl;
    }

}

#endif
