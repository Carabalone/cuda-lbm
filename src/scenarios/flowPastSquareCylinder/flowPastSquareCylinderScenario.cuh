#ifndef FLOW_PAST_SQUARE_CYLINDER_SCENARIO_H
#define FLOW_PAST_SQUARE_CYLINDER_SCENARIO_H

#include "scenarios/scenario.cuh"
#include "functors/initialConditions/defaultInit.cuh"
#include "IBM/config/body_config.hpp"
#include "scenarios/flowPastSquareCylinder/FlowPastSquareCylinderFunctors.cuh"

struct FlowPastSquareCylinderScenario : public ScenarioTrait<
    DefaultInit<3>,
    FlowPastSquareCylinderBoundary,
    void,
    CM<3, NoAdapter>
> {
    static constexpr float u_max = 0.05f;
    static constexpr float D = 16.0f;
    static constexpr float Re = 1000.0f;
    static constexpr float viscosity = u_max * D / Re;
    static constexpr float tau = viscosity_to_tau(viscosity);
    static constexpr float omega = 1.0f / tau;
    static constexpr int ramp_duration = 1000;

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

    static const char* name() { return "Flow Past Square Cylinder (Regularized Inlet)"; }

    static InitType init() { return InitType(); }
    static BoundaryType boundary() { return BoundaryType(); }
    static ValidationType validation() { return ValidationType(); }

    template <typename LBMSolver>
    static float compute_error(LBMSolver& solver) {
        return -1.0f;
    }
};

#endif
