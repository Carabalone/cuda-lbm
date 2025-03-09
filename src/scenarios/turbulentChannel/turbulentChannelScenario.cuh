#ifndef TURBULENT_CHANNEL_SCENARIO_H
#define TURBULENT_CHANNEL_SCENARIO_H

#include "../scenario.cuh"
#include "turbulentChannelFunctors.cuh"

struct TurbulentChannelScenario : public ScenarioTrait <
    TurbulentChannelInit,
    TurbulentChannelBoundary,
    TurbulentChannelValidation
> {
    static constexpr float Re = 3000.0f;
    static constexpr int N_half = NY / 2;
    
    static constexpr float u_max = 0.1f;
    static constexpr float viscosity = u_max * N_half / Re;
    static constexpr float tau = viscosity_to_tau(viscosity);
    static constexpr float omega = 1.0f / tau;
    
    static constexpr float perturbation = 0.05f * u_max;
    
    static const char* name() { return "TurbulentChannel"; }
    
    static InitType init() { 
        return InitType(u_max, perturbation); 
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