#ifndef FLOW_PAST_CYLINDER_SCENARIO_H
#define FLOW_PAST_CYLINDER_SCENARIO_H

#include "scenarios/scenario.cuh"
#include "flowPastCylinderFunctors.cuh"
#include "functors/includes.cuh"

struct FlowPastCylinderScenario : public ScenarioTrait <
    FlowPastCylinderInit,
    FlowPastCylinderBoundary,
    FlowPastCylinderValidation,
    BGK
    // CM<NoAdapter>
> {
    static constexpr float Re = 50.0f;             
    
    static constexpr float D = 16.0f;               
    static constexpr float r = D/2.0f;             
    
    static constexpr float cx = 48.0f;              // ~3D from inlet
    static constexpr float cy = NY/2.0f;            
    
    static constexpr float u_max = 0.05f;           
    static constexpr float viscosity = u_max * D / Re; 
    static constexpr float tau = viscosity_to_tau(viscosity);
    static constexpr float omega = 1.0f / tau;
    static float current_velocity;
    static constexpr bool has_analytical_solution = false;
    
    // central moments deifnition have a different moment order
    static constexpr float S[quadratures] = {
        0.0f,      // ρ (density) - conserved
        0.0f,      // kₓ (first-order x central moment - conserved) 
        0.0f,      // kᵧ (first-order y central moment - conserved)
        1.0f,     // kₓₓ + kᵧᵧ (bulk viscosity)
        omega,     // kₓₓ - kᵧᵧ (shear viscosity)
        omega,     // kₓᵧ (shear viscosity)
        1.0f,     // kₓₓᵧ (higher-order)
        1.0f,     // kₓᵧᵧ (higher-order)
        1.0f      // kₓₓᵧᵧ (higher-order)
    };

    static void add_bodies() {
        IBM_bodies.push_back(create_cylinder(cx, cy, r));
    }
    // static constexpr float S[quadratures] = {
    //     0.0f,  // density (conserved)
    //     1.2f, // bulk viscosity related - controls compressibility
    //     1.1f, // energy flux tensor
    //     0.0f,  // momentum-x (conserved)
    //     1.2f, // energy square moment - affects stability in high Reynolds number flows
    //     0.0f,  // momentum-y (conserved)
    //     1.4f, // third-order moment - affects numerical stability near boundaries
    //     omega, // kinematic viscosity (shear viscosity)
    //     omega  // kinematic viscosity (shear viscosity)
    // };
    
    static const char* name() { return "FlowPastCylinder"; }
    
    static InitType init() { 
        return InitType(u_max, cx, cy, r); 
    }

    static BoundaryType boundary() {
        return BoundaryType(cx, cy, r);
    }
    
    static ValidationType validation() {
        return ValidationType();
    }
    
    template <typename LBMSolver>
    static float compute_error(LBMSolver& solver) {
        return -1.0f;
    }
};

#endif // !FLOW_PAST_CYLINDER_SCENARIO_H