// #include <iostream>
// #include <fstream>
// #include <sstream>
// #include <vector>
// #include <string>
// #include <cstdlib>

// #include "third_party/nanoflann/nanoflann.hpp"
// #include "utils.hpp"
// #include "sampler.hpp"
// #include <iomanip>

// namespace sm = sampler;

// int main(int argc, char** argv) {
//     const int num_initial_points = 10000;
//     const float sphere_radius = 10.0f;
//     const int target_samples = 5000;

//     float area = 4.0f * static_cast<float>(M_PI) * sphere_radius * sphere_radius;
//     float r_max = std::sqrt(area / (2.0f * std::sqrt(3.0f) * static_cast<float>(target_samples)));

//     std::cout << std::fixed << std::setprecision(4);
//     std::cout << "--- Configuration ---" << std::endl;
//     std::cout << "Initial points: " << num_initial_points << std::endl;
//     std::cout << "Sphere radius: " << sphere_radius << std::endl;
//     std::cout << "Target final samples: " << target_samples << std::endl;
//     std::cout << "Calculated r_max for sampler: " << r_max << std::endl;
//     std::cout << "---------------------" << std::endl;

//     // std::vector<Point3> vertices = generate_sphere_points(num_initial_points, sphere_radius);
//     std::vector<geom::Point3D> vertices;
//     if (argc == 2) {
//         std::cout << "Loading mesh from: " << argv[1] << std::endl;
//         sm::MeshData mesh = sm::load_mesh_from_obj(argv[1]);
    
//         if (mesh.vertices.empty()) {
//             std::cerr << "Failed to load mesh or mesh is empty. Aborting."
//                       << std::endl;
//             return EXIT_FAILURE;
//         }
//         vertices = mesh.vertices;
//         if (mesh.faces.empty()) {
//             std::cerr << "No faces in mesh. Aborting. " << std::endl;
//             return EXIT_FAILURE;
//         }

//         double volume = calculate_mesh_volume(mesh.vertices, mesh.faces);
//         std::cout << "Calculated Mesh Volume (A3): " << volume
//                     << std::endl;
//         r_max = sm::calculate_r_max_3d(volume, target_samples);
//     }
//     else if (argc == 1) {
//         vertices = sm::generate_sphere_points(
//             num_initial_points, sphere_radius
//         );
//     }
//     else {
//         std::cerr << "Usage: " << argv[0]
//                   << " [optional_input.obj]\n";
//         return EXIT_FAILURE;
//     }


//     save_points_to_csv(vertices, "init_vertices.csv");

//     std::vector<sm::Sample3> samples = sm::Sample3::from_points(vertices);
//     sm::Sampler sampler(target_samples, r_max, samples);

//     std::vector<geom::Point3D> sampled = sampler.eliminate_samples();

//     save_points_to_csv(sampled, "sampled_vertices.csv");

//     return 0;
// }