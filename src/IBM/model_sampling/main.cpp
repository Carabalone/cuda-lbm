#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "third_party/nanoflann/nanoflann.hpp"
#include "IBM/model_sampling/points.hpp"
#include "utils.hpp"
#include "sampler.hpp"
#include <iomanip>

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


int main(void) {
    const int num_initial_points = 5000;
    const float sphere_radius = 10.0f;
    const int target_samples = 500;

    float area = 4.0f * static_cast<float>(M_PI) * sphere_radius * sphere_radius;
    float r_max = std::sqrt(area / (2.0f * std::sqrt(3.0f) * static_cast<float>(target_samples)));

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "--- Configuration ---" << std::endl;
    std::cout << "Initial points: " << num_initial_points << std::endl;
    std::cout << "Sphere radius: " << sphere_radius << std::endl;
    std::cout << "Target final samples: " << target_samples << std::endl;
    std::cout << "Calculated r_max for sampler: " << r_max << std::endl;
    std::cout << "---------------------" << std::endl;

    std::vector<Point3> vertices = generate_sphere_points(num_initial_points, sphere_radius);
    save_points_to_csv(vertices, "init_vertices.csv");

    std::vector<Sample3> samples = Sample3::from_points(vertices);
    Sampler sampler(target_samples, r_max, samples);

    std::vector<Point3> sampled = sampler.eliminate_samples();

    save_points_to_csv(sampled, "sampled_vertices.csv");

    return 0;
}