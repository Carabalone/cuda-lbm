#ifndef FLOW_SCENARIO_H
#define FLOW_SCENARIO_H

#include <vector>
#include <array>
#include <string>
#include "core/collision.cuh"

#define DEFAULT_MRT_S_MATRIX(omega_val) { \
    0.0f,        /* density (conserved) */ \
    omega_val,   /* bulk viscosity */ \
    omega_val,   /* energy flux tensor */ \
    0.0f,        /* momentum-x (conserved) */ \
    omega_val,   /* energy square moment */ \
    0.0f,        /* momentum-y (conserved) */ \
    omega_val,   /* third-order moment */ \
    omega_val,   /* kinematic viscosity */ \
    omega_val    /* kinematic viscosity */ \
}

template <typename InitFunctor, typename BoundaryFunctor, typename ValidationFunctor = void, typename CollisionType = BGK>
struct ScenarioTrait {
    using InitType = InitFunctor;
    using BoundaryType = BoundaryFunctor;
    using ValidationType = ValidationFunctor;
    using CollisionOp = CollisionType;
    
    static constexpr float viscosity = 1.0f/6.0f;
    static constexpr float tau = viscosity_to_tau(viscosity);
    static constexpr float omega = 1.0f / tau;
    static inline float t = 0.0f;

    // WARNING: PLEASE DEFINE YOUR OWN S MATRIX IF YOU ARE GOING TO USE MRT IN A SCENARIO.
    // IN THE ABSENSE OF AN OVERWRITE, THE PROGRAM DEFAULTS TO THIS S MATRIX WHICH WILL MATCH
    // BGK WITH A TAU AND OMEGA OF 1.0

    // all non-conserved moments set to viscosity to match a BGK implementation 
    static constexpr float S[quadratures] = {
        0.0f,      // density (conserved)
        omega, // bulk viscosity related - controls compressibility
        omega, // energy flux tensor
        0.0f,      // momentum-x (conserved)
        omega, // energy square moment - affects stability in high Reynolds number flows
        0.0f,      // momentum-y (conserved)
        omega, // third-order moment - affects numerical stability near boundaries
        omega, // kinematic viscosity (shear viscosity)
        omega  // kinematic viscosity (shear viscosity)
    };

    static constexpr float u_max = 0.1f;
    
    static const char* name() { return "BaseScenario"; }
    
    static InitType initCondition() { return InitType(); }
    static BoundaryType boundaryCondition() { return BoundaryType(); }
    
    static constexpr bool has_analytical_solution = !std::is_same<ValidationFunctor, void>::value;
    
    template <typename LBMSolver>
    static float compute_error(LBMSolver& solver);

    static void update_ts(float new_ts) {
        t = new_ts;
    }
};

#endif // ! FLOW_SCENARIO_H