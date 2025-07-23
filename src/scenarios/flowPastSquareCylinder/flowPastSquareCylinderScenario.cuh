#ifndef FLOW_PAST_SQUARE_CYLINDER_H
#define FLOW_PAST_SQUARE_CYLINDER_H

#include "scenarios/scenario.cuh"
#include "functors/initialConditions/defaultInit.cuh"
#include "IBM/config/body_config.hpp"

struct FlowPastSquareCylinderBoundary {

    __host__ __device__ int operator()(int x, int y, int z) {

        const int edges = (x==0) + (x==NX-1) +
                          (y==0) + (y==NY-1) +
                          (z==0) + (z==NZ-1);

        if (edges >= 2) return BC_flag::CORNER_EDGE_BOUNCE_BACK;

        if (x == 0)
            return BC_flag::REGULARIZED_INLET_LEFT;
            // return BC_flag::GUO_VELOCITY_INLET;
            // return BC_flag::BOUNCE_BACK;

        if (x == NX - 1) return BC_flag::GUO_PRESSURE_OUTLET;


        if (y == 0 || y == NY-1 || z == 0 || z == NZ-1)
            return BC_flag::BOUNCE_BACK;

        return BC_flag::FLUID;
    }

};

struct FlowInit {

    __host__ __device__
    FlowInit() { }

    __device__
    inline void apply_forces(float* rho, float* u, float* force, int node) {

        force[get_vec_index(node, 0)] = 0.0f;
        force[get_vec_index(node, 1)] = 0.0f;
        force[get_vec_index(node, 2)] = 0.0f;

    }

     __device__
    inline void operator()(float* rho, float* u, float* force, int node) {
        rho[node] = 1.0f;

        u[get_vec_index(node, 0)] = 0.05f;
        u[get_vec_index(node, 1)] = 0.0f;
        u[get_vec_index(node, 2)] = 0.0f;

        apply_forces(rho, u, force, node);
    }
};

struct FlowPastSquareCylinderScenario : public ScenarioTrait<
    // DefaultInit<3>,
    FlowInit,
    FlowPastSquareCylinderBoundary,
    void,
    CM<3, NoAdapter>
> {

    static constexpr float D         = 16.0f;
    static constexpr float Re        = 1000.0f;
    static constexpr float u_max     = 0.05f;
    static constexpr float viscosity = u_max * D / Re;
    static constexpr float tau       = viscosity_to_tau(viscosity);
    static constexpr float omega     = 1.0f / tau;

    static void add_bodies() {
        
        // std::string base_path = "";
        // #ifdef _WIN32
        //     base_path = "../../../";
        // #endif
        // std::string sphere_config = base_path + "src/scenarios/flowPastSquareCylinder/sphere_config.ini";
        // IBMBody sphere = conf::create_body_from_config(sphere_config);
        // IBM_bodies.emplace_back(sphere);

    }

    static constexpr float S[quadratures] = {
        1.0f, 1.0f, 1.0f, 1.0f, omega, omega, omega, omega, omega,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
    };

    static const char* name() { return "FlowPastSquareCylinder"; }

    static InitType      init()      { return InitType(); }
    static BoundaryType  boundary()  { return BoundaryType(); }
    static ValidationType validation(){ return ValidationType(); }
};

#endif //! FLOW_PAST_SQUARE_CYLINDER_H
