#ifndef FLOW_SCENARIO_H
#define FLOW_SCENARIO_H

#include <vector>
#include <array>
#include <string>

template <typename InitFunctor, typename BoundaryFunctor, typename ValidationFunctor = void>
struct ScenarioTrait {
    using InitType = InitFunctor;
    using BoundaryType = BoundaryFunctor;
    using ValidationType = ValidationFunctor;
    
    static constexpr float viscosity = 1.0f/6.0f;
    static constexpr float tau = viscosity_to_tau(viscosity);
    static constexpr float omega = 1.0f / tau;
    static inline float t = 0.0f;
    
    static const char* name() { return "BaseScenario"; }
    
    static InitType initCondition() { return InitType(); }
    static BoundaryType boundaryCondition() { return BoundaryType(); }
    
    static constexpr bool has_analytical_solution = !std::is_same<ValidationFunctor, void>::value;
    
    template <typename LBMSolver>
    static float computeError(LBMSolver& solver);

    static void update_ts(float new_ts) {
        t = new_ts;
    }
};

#endif // ! FLOW_SCENARIO_H