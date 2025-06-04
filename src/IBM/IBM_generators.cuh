#ifndef IBM_GENERATORS_H
#define IBM_GENERATORS_H

#include "IBM/IBMBody.cuh"
#include "IBM/geometry/point.hpp"

// assumes grid coordinates (system, but doesn't need to be aligned)
IBMBody create_cylinder(float cx, float cy, float r, int num_pts=16);
IBMBody create_sphere(float cx, float cy, float cz, float r, int n_theta, int n_phi);
IBMBody load_from_obj(const std::string& filename, int target_samples=-1); // 3d for now
IBMBody load_from_csv(const std::string& csv_filename);

template <int dim>
IBMBody body_from_points(const std::vector<geom::Point<dim>>& input_points) {

    IBMBody body;
    body.num_points = static_cast<int>(input_points.size());

    body.points = new float[dim * body.num_points];
    body.velocities = new float[dim * body.num_points];


    for (int i = 0; i < body.num_points; i++) {
        for (int comp = 0; comp < dim; comp++) {
            body.points[i * dim + comp]     = input_points[i].coords[comp];
            body.velocities[i * dim + comp] = 0.0f;
        }
    }
    return body;
}


#endif // !IBM_GENERATORS_H
