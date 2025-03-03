#ifndef TAYLOR_GREEN_SCENARIO_H
#define TAYLOR_GREEN_SCENARIO_H

#include "../scenario.cuh"
#include "taylorGreenFunctors.cuh"

struct TaylorGreenScenario : public ScenarioTrait <
    TaylorGreenInit,
    TaylorGreenBoundary,
    TaylorGreenValidation
> {
    static constexpr float u0 = 0.04f;
    static constexpr float viscosity = 1.0f/6.0f;
    static constexpr float tau = viscosity_to_tau(viscosity);
    static constexpr float omega = 1.0f / tau;
    
    static const char* name() { return "TaylorGreen"; }
    
    static InitType init() { 
        return InitType(viscosity, u0); 
    }

    static BoundaryType boundary() {
        return BoundaryType();
    }
    
    static ValidationType analyticalSolution() {
        return ValidationType(u0, viscosity, t);
    }
    
    template <typename LBMSolver>
    static float computeError(LBMSolver& solver) {
        auto analytical_field = analyticalSolution().getFullField();
        
        if (solver.update_ts < solver.timestep)
            solver.update_macroscopics();
        
        float error_sum = 0.0f;
        float velocity_norm_sum = 0.0f;
        
        for (int y = 0; y < NY; y++) {
            for (int x = 0; x < NX; x++) {
                int index = (y * NX + x) * dimensions;
                
                float ux_sim = solver.h_u[index];
                float uy_sim = solver.h_u[index + 1];
                
                float ux_ana = analytical_field[y * NX + x][0];
                float uy_ana = analytical_field[y * NX + x][1];
                
                float diff_x = ux_sim - ux_ana;
                float diff_y = uy_sim - uy_ana;
                
                error_sum += diff_x * diff_x + diff_y * diff_y;
                velocity_norm_sum += ux_ana * ux_ana + uy_ana * uy_ana;
            }
        }
        
        return std::sqrt(error_sum / velocity_norm_sum) * 100.0f;
    }
};

#endif // TAYLOR_GREEN_SCENARIO_H