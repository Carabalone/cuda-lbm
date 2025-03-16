#ifndef TAYLOR_GREEN_SCENARIO_H
#define TAYLOR_GREEN_SCENARIO_H

#include "scenarios/scenario.cuh"
#include "taylorGreenFunctors.cuh"

struct TaylorGreenScenario : public ScenarioTrait <
    TaylorGreenInit,
    TaylorGreenBoundary,
    TaylorGreenValidation,
    CM
> {
    static constexpr float u_max = 0.04f;
    static constexpr float viscosity = 1.0f/6.0f;
    static constexpr float tau = viscosity_to_tau(viscosity);
    static constexpr float omega = 1.0f / tau;

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
    
    static const char* name() { return "TaylorGreen"; }
    
    static InitType init() { 
        return InitType(viscosity, u_max); 
    }

    static BoundaryType boundary() {
        return BoundaryType();
    }
    
    static ValidationType validation() {
        return ValidationType(u_max, viscosity, t);
    }
    
    template <typename LBMSolver>
    static float compute_error(LBMSolver& solver) {
        auto analytical_field = validation().getFullField();
        
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