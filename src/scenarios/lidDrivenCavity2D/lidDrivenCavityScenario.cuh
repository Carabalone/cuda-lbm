#ifndef LID_DRIVEN_SCENARIO_H
#define LID_DRIVEN_SCENARIO_H

#include "../scenario.cuh"
#include "lidDrivenCavityFunctors.cuh"


struct LidDrivenScenario : public ScenarioTrait <
    LidDrivenInit,
    LidDrivenBoundary,
    LidDrivenValidation
> {
    // for Re = 100, tau = 1.2, L = 128 = NX - 1 
    // L = 128 = NX - 1 
    static constexpr float u_max =  0.0517f;
    static constexpr float viscosity = 0.0667f; 
    static constexpr float tau = viscosity_to_tau(viscosity);
    static constexpr float omega = 1.0f / tau;
    
    static const char* name() { return "LidDriven"; }
    
    static InitType init() { 
        return InitType(u_max); 
    }

    static BoundaryType boundary() {
        return BoundaryType();
    }
    
    static ValidationType validation() {
        return ValidationType();
    }
    
    template <typename LBMSolver>
    static float computeError(LBMSolver& solver) {

        if (solver.update_ts < solver.timestep)
            solver.update_macroscopics();

        return -1.0f;
    }
};

#endif // !LID_DRIVEN_SCENARIO_H