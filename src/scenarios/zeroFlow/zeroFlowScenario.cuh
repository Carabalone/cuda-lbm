#ifndef ZERO_FLOW_BOX_SCENARIO_H
#define ZERO_FLOW_BOX_SCENARIO_H

#include "scenarios/scenario.cuh"
#include "functors/initialConditions/defaultInit.cuh"
#include "IBM/config/body_config.hpp"

struct ZeroFlowBoundary {
    __host__ __device__ int operator()(int x, int y, int z) {

        if (y == 0 || y == NY - 1 || z == 0 || z == NZ - 1)
            return BC_flag::BOUNCE_BACK;

        if (x == 0)      return BC_flag::REGULARIZED_INLET_LEFT;
        if (x == NX - 1) return BC_flag::PRESSURE_OUTLET;
        // if (x == NX - 1) return BC_flag::ZG_OUTFLOW;

        return BC_flag::FLUID;
    }
};

struct ZeroFlowScenario : public ScenarioTrait<
    DefaultInit<3>,
    ZeroFlowBoundary,
    void,
    CM<3, NoAdapter>
> {

    static constexpr float u_max     = 0.05f;
    static constexpr float viscosity = 1.0f/6.0f;
    static constexpr float tau       = viscosity_to_tau(viscosity);
    static constexpr float omega     = 1.0f / tau;

    static void add_bodies() {
        
        std::string base_path = "";
        #ifdef _WIN32
            base_path = "../../../";
        #endif
        std::string sphere_config = base_path + "src/scenarios/zeroFlow/sphere_config.ini";
        IBMBody sphere = conf::create_body_from_config(sphere_config);
        // IBM_bodies.emplace_back(sphere);

    }

    static constexpr float S[quadratures] = {
        1.0f, 1.0f, 1.0f, 1.0f, omega, omega, omega, omega, omega,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
    };

    static const char* name() { return "Zero-Flow Periodic Box"; }

    static InitType      init()      { return InitType(); }
    static BoundaryType  boundary()  { return BoundaryType(); }
    static ValidationType validation(){ return ValidationType(); }
};

#endif
