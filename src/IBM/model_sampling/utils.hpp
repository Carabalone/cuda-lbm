#pragma once
#include "points.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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