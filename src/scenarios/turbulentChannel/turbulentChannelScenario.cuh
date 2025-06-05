#ifndef TURBULENT_CHANNEL_SCENARIO_H
#define TURBULENT_CHANNEL_SCENARIO_H

#include "scenarios/scenario.cuh"
#include "turbulentChannelFunctors.cuh"
#include "IBM/IBMUtils.cuh"
#include "IBM/mesh/mesh_io.hpp"
#include "IBM/mesh/mesh_transformer.hpp"
#include "IBM/config/body_config.hpp"

struct TurbulentChannelScenario : public ScenarioTrait <
    TurbulentChannelInit,
    TurbulentChannelBoundary,
    TurbulentChannelValidation,
    CM<3, NoAdapter>
> {
    // static constexpr float Re = 3300.0f;
    // static constexpr float Re_tau = 180.0f;
    static constexpr int N_half = NY / 2;
    
    // static constexpr float u_max = 0.1f;
    static constexpr float u_tau = 0.00579f; // Mach number of 0.01
    // static constexpr float viscosity = u_tau * N_half / Re_tau;
    // static constexpr float u_max = viscosity * Re / N_half;
    // static constexpr float u_tau = viscosity * Re_tau / N_half;
    static constexpr float Re = 200001;
    static constexpr float u_max = 0.1f;
    static constexpr float viscosity = u_max * N_half / Re;
    static constexpr float tau = viscosity_to_tau(viscosity);
    static constexpr float omega = 1.0f / tau;

    static constexpr float perturbation = 0.1f;

    static void add_bodies() {
        
        float cx = NX / 4.0f;
        float cy = NY / 2.0f;
        float cz = NZ / 2.0f;
        float r  = NY / 4.0f;

        // IBMBody sphere = create_sphere(cx, cy, cz, r, 64, 64);
        // IBM_bodies.emplace_back(sphere);

        // IBM_bodies.emplace_back(load_from_obj("assets/horse.obj", 360));
        std::string base_path = "";
        #ifdef _WIN32
            base_path = "../../../";
        #endif

        // std::string obj_path = base_path + "/sphere_unit_64x64.obj";

        // mesh::MeshData sphere_mesh = mesh::load_obj(obj_path);
        // if (sphere_mesh.vertices.empty()) {
        //     std::cerr << "[add_bodies] Failed to load sphere mesh from: " << obj_path << std::endl;
        //     return;
        // }

        // mesh::MeshTransformer transformer(sphere_mesh);
        // transformer
        //     .scale_to_overall_size(r * 2.0f)
        //     .rotate_y(static_cast<float>(M_PI) / 4.0f)
        //     .rotate_x(static_cast<float>(M_PI) / 6.0f)
        //     .move_anchor_to_world(cx, cy, cz);

        // transformer.collect_file("debug_transformed_sphere_test.obj");

        // IBMBody sphere_body = transformer.collect_ibm_body(1024);
        // IBM_bodies.push_back(sphere_body);

        std::string sphere_config = base_path + "src/scenarios/turbulentChannel/sphere_body_config.ini";
        IBMBody sphere = conf::create_body_from_config(sphere_config);
        IBM_bodies.emplace_back(sphere);

        
    }

    static constexpr float S[quadratures] = {
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        omega,
        omega,
        omega,
        omega,
        omega,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f
    };
    
    // // Lallemand and Luo (2000) magic parameters.
    // static constexpr float S[quadratures] = {
    //     0.0f,  // density (conserved)
    //     1.63f, // bulk viscosity related - controls compressibility
    //     1.14f, // energy flux tensor
    //     0.0f,  // momentum-x (conserved)
    //     1.92f, // energy square moment - affects stability in high Reynolds number flows
    //     0.0f,  // momentum-y (conserved)
    //     1.98f, // third-order moment - affects numerical stability near boundaries
    //     omega, // kinematic viscosity (shear viscosity)
    //     omega  // kinematic viscosity (shear viscosity)
    // };
    // static constexpr float S[quadratures] = {0.0f};
    
    static const char* name() { return "TurbulentChannel"; }
    
    static InitType init() { 
        return InitType(u_max, u_tau, perturbation); 
    }

    static BoundaryType boundary() {
        return BoundaryType();
    }
    
    static ValidationType validation() {
        return ValidationType();
    }
    
    template <typename LBMSolver>
    static float compute_error(LBMSolver& solver) {
        if (solver.update_ts < solver.timestep)
            solver.update_macroscopics();

        return -1.0f;
    }
};

#endif // !TURBULENT_CHANNEL_SCENARIO_H
