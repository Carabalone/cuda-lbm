#include "IBM/IBM_generators.cuh"

IBMBody create_cylinder(float cx, float cy, float r, int num_pts) {

    IBMBody body = {num_pts, nullptr, nullptr};
    body.points = new float[2*num_pts];
    body.velocities = new float[2*num_pts];

    float angle = 2 * M_PI / num_pts; float coord[2];
    for (int i=0; i < num_pts; i++) {
        coord[0] = cx + r*cos(i*angle);
        coord[1] = cy + r*sin(i*angle);

        body.points[2*i]   = coord[0];
        body.points[2*i+1] = coord[1];
        
        body.velocities[2*i]   = 0.0f;
        body.velocities[2*i+1] = 0.0f;
    }


    return body;
}

IBMBody create_sphere(float cx, float cy, float cz, float r, int n_theta, int n_phi) {
    if (n_phi < 2) n_phi = 2;
    if (n_theta < 3) n_theta = 3; // need at least a triangle

    int num_pts = 2 + (n_phi - 2) * n_theta;
    IBMBody body { num_pts, nullptr, nullptr };
    body.points    = new float[3 * num_pts];
    body.velocities= new float[3 * num_pts];

    float dtheta = 2.0f * M_PI / n_theta;
    float dphi   = M_PI    / (n_phi - 1);

    int idx = 0;
    // 1) North pole (φ=0)
    body.points[3*idx + 0] = cx;
    body.points[3*idx + 1] = cy + r;
    body.points[3*idx + 2] = cz;
    body.velocities[3*idx + 0] = 0;
    body.velocities[3*idx + 1] = 0;
    body.velocities[3*idx + 2] = 0;
    ++idx;

    // 2) Intermediate rings φ in (dphi, (n_phi-2)*dphi)
    for (int i = 1; i < n_phi - 1; ++i) {
        float phi = i * dphi;
        float sin_phi = sinf(phi);
        float cos_phi = cosf(phi);
        for (int j = 0; j < n_theta; ++j) {
            float theta = j * dtheta;
            float cos_theta = cosf(theta), sin_theta = sinf(theta);

            float x = cx + r * sin_phi * cos_theta;
            float y = cy + r * cos_phi;
            float z = cz + r * sin_phi * sin_theta;

            body.points[3*idx + 0] = x;
            body.points[3*idx + 1] = y;
            body.points[3*idx + 2] = z;
            body.velocities[3*idx + 0] = 0;
            body.velocities[3*idx + 1] = 0;
            body.velocities[3*idx + 2] = 0;
            ++idx;
        }
    }

    // 3) South pole (φ=π)
    body.points[3*idx + 0] = cx;
    body.points[3*idx + 1] = cy - r;
    body.points[3*idx + 2] = cz;
    body.velocities[3*idx + 0] = 0;
    body.velocities[3*idx + 1] = 0;
    body.velocities[3*idx + 2] = 0;
    // idx == num_pts-1 here

    return body;
}

