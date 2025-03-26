#include "IBM/IBMBody.cuh"

IBMBody create_cylinder(float cx, float cy, float r, int num_pts) {

    IBMBody body = {num_pts, nullptr, nullptr};
    body.points = new float[2*num_pts];
    body.velocities = new float[2*num_pts];

    float angle = 2*M_PI / num_pts; float coord[2];
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