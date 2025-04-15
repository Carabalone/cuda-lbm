#ifndef TAYLOR_GREEN_3D_SCENARIO_H
#define TAYLOR_GREEN_3D_SCENARIO_H

#include "scenarios/scenario.cuh"
#include "taylorGreen3DFunctors.cuh"

struct TaylorGreen3DScenario : public ScenarioTrait <
    TaylorGreen3DInit,
    TaylorGreen3DBoundary,
    TaylorGreen3DValidation,
    BGK<3>
> {
    static constexpr float u_max = 0.04f;
    static constexpr float viscosity = 1.0f/6.0f;
    static constexpr float tau = viscosity_to_tau(viscosity);
    static constexpr float omega = 1.0f / tau;
    
    static const char* name() { return "TaylorGreen3D"; }
    
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
        
        for (int z = 0; z < NZ; z++) {
            for (int y = 0; y < NY; y++) {
                for (int x = 0; x < NX; x++) {
                    int node_index = z * (NX * NY) + y * NX + x;
                    
                    float ux_sim = solver.h_u[get_vec_index(node_index, 0)];
                    float uy_sim = solver.h_u[get_vec_index(node_index, 1)];
                    float uz_sim = solver.h_u[get_vec_index(node_index, 2)];
                    
                    float ux_ana = analytical_field[node_index][0];
                    float uy_ana = analytical_field[node_index][1];
                    float uz_ana = analytical_field[node_index][2];
                    
                    float diff_x = ux_sim - ux_ana;
                    float diff_y = uy_sim - uy_ana;
                    float diff_z = uz_sim - uz_ana;
                    
                    error_sum += diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
                    velocity_norm_sum += ux_ana * ux_ana + uy_ana * uy_ana + uz_ana * uz_ana;
                }
            }
        }
        
        return std::sqrt(error_sum / velocity_norm_sum) * 100.0f;
    }
};

#endif // TAYLOR_GREEN_3D_SCENARIO_H