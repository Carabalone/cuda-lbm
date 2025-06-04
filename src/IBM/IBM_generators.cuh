#ifndef IBM_GENERATORS_H
#define IBM_GENERATORS_H

#include "IBM/IBMBody.cuh"

// assumes grid coordinates (system, but doesn't need to be aligned)
IBMBody create_cylinder(float cx, float cy, float r, int num_pts=16);
IBMBody create_sphere(float cx, float cy, float cz, float r, int n_theta, int n_phi);
IBMBody load_from_obj(const std::string& filename, int target_samples=-1); // 3d for now
IBMBody load_from_csv(const std::string& csv_filename);

#endif // !IBM_GENERATORS_H
