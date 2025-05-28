#include "sampler.hpp"

namespace sampler {

    float calculate_mesh_volume(const std::vector<Point3>& vertices,
                                const std::vector<Face>& faces) {
        if (vertices.empty() || faces.empty()) {
            std::cerr << "[WARNING] Cannot calculate volume for empty or faceless mesh." << std::endl;
            return 0.0f;
        }

        double volume_acc = 0.0;

        for (const auto& face : faces) {
            if (face.v_indices[0] < 0 || face.v_indices[0] >= vertices.size() ||
                face.v_indices[1] < 0 || face.v_indices[1] >= vertices.size() ||
                face.v_indices[2] < 0 || face.v_indices[2] >= vertices.size()) {
                std::cerr << "[WARNING] Invalid vertex index in face. Skipping face." << std::endl;
                continue;
            }

            const Point3& p1 = vertices[face.v_indices[0]];
            const Point3& p2 = vertices[face.v_indices[1]];
            const Point3& p3 = vertices[face.v_indices[2]];

            // Calculate signed volume of the tetrahedron (p1, p2, p3, origin)
            // V_tet = (1/6) * dot(p1, cross(p2, p3))
            // cross(p2, p3) = (p2.y*p3.z - p2.z*p3.y, p2.z*p3.x - p2.x*p3.z, p2.x*p3.y - p2.y*p3.x)
            
            double v321 = static_cast<double>(p3.x) * p2.y * p1.z;
            double v231 = static_cast<double>(p2.x) * p3.y * p1.z;
            double v312 = static_cast<double>(p3.x) * p1.y * p2.z;
            double v132 = static_cast<double>(p1.x) * p3.y * p2.z;
            double v213 = static_cast<double>(p2.x) * p1.y * p3.z;
            double v123 = static_cast<double>(p1.x) * p2.y * p3.z;

            volume_acc += (-v321 + v231 + v312 - v132 - v213 + v123);
        }
        return static_cast<float>(std::abs(volume_acc / 6.0));
    }

    float calculate_r_max_3d(float domain_volume, int target_samples) {
        if (domain_volume <= 1e-6f || target_samples <= 0) {
            return 0.0f;
        }
        return std::cbrt(domain_volume /
                        (4.0f * std::sqrt(2.0f) *
                        static_cast<float>(target_samples)));
    }

}