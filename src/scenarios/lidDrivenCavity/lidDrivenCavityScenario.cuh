#ifndef LID_DRIVEN_SCENARIO_H
#define LID_DRIVEN_SCENARIO_H

#include "scenarios/scenario.cuh"
#include "lidDrivenCavityFunctors.cuh"
#include "core/collision.cuh"


struct LidDrivenScenario : public ScenarioTrait <
    LidDrivenInit,
    LidDrivenBoundary,
    LidDrivenValidation,
    CM
> {
    // Re=100
    // static constexpr float u_max =  0.0517f;
    // static constexpr float viscosity = 0.0667f; 

    // Re=1000
    // static constexpr float u_max =  0.1f;
    // static constexpr float viscosity = 0.0128f;

    // Re=3200 (innacurate, but stable) (err ~= 25%) (already unstable for non-regularized boundaries + BGK)
    // stable for non-regularized boundaries + MRT
    // stable for anything with regularized boundaries
    // static constexpr float u_max =  0.1f;
    // static constexpr float viscosity = 0.004f;
    // static constexpr float viscosity = 0.0064f; // for 2000


    // Re=5000 (stable & accurate @ 3.72% error)
    // static constexpr float u_max =  0.1f;
    // static constexpr float viscosity = 0.00258f;

    // Re=7500
    // static constexpr float u_max =  0.1f;
    // static constexpr float viscosity = 0.00172f;

    // Re=10_000
    static constexpr float u_max =  0.1f;
    static constexpr float viscosity = 0.00128f;
    

    static constexpr float tau = viscosity_to_tau(viscosity);
    static constexpr float omega = 1.0f / tau;

    // CMs
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

    // MRT
    // static constexpr float S[quadratures] = {
    //     0.0f,      // density (conserved)
    //     1.0f, // bulk viscosity related - controls compressibility
    //     1.4f, // energy flux tensor
    //     0.0f,      // momentum-x (conserved)
    //     1.2f, // energy square moment - affects stability in high Reynolds number flows
    //     0.0f,      // momentum-y (conserved)
    //     1.9f, // third-order moment - affects numerical stability near boundaries
    //     omega, // kinematic viscosity (shear viscosity)
    //     omega  // kinematic viscosity (shear viscosity)
    // };
    
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
    static float compute_error(LBMSolver& solver) {

        if (solver.update_ts < solver.timestep)
            solver.update_macroscopics();

        const auto validator = validation();
        const int re = compute_reynolds(u_max, NY, viscosity);

        float error_sum_ux = 0.0f;
        float ref_sum_ux = 0.0f;
        float error_sum_uy = 0.0f;
        float ref_sum_uy = 0.0f;
        
        int center_x = NX / 2;
        int center_y = NY / 2;
        
        // printf("\n%-8s %-8s %-10s %-10s %-10s | %-8s %-10s %-10s %-10s\n", 
        //       "y", "node_y", "ux-LBM", "ux-Ghia", "ux-Diff", 
        //       "node_x", "uy-LBM", "uy-Ghia", "uy-Diff");
        // printf("---------------------------------------------------------------------------------\n");
        const float* ux_ref = validator.get_closest_ref_data(re, true);
        const float* uy_ref = validator.get_closest_ref_data(re, false);
        
        for (int i = 0; i < validator.ghia_u_count; i++) {
            // processing ux along mid y axis. (y_norm = 0.5)
            float y_norm = validator.ghia_y[i];
            float ux_ghia = ux_ref[i];
            
            int y = round(y_norm * (NY - 1));
            y = std::max(0, std::min(y, NY-1));
            
            int node_ux = y * NX + center_x;
            float ux_lbm = solver.h_u[2*node_ux] * (1.0f/u_max); // scale for ghias u_max = 1.0f
            
            float diff_ux = ux_lbm - ux_ghia;
            error_sum_ux += diff_ux * diff_ux;
            ref_sum_ux += ux_ghia * ux_ghia;
            
            // processing uy along mid x axis. (x_norm = 0.5)
            float x_norm = validator.ghia_x[i];
            float uy_ghia = uy_ref[i];
            
            int x = round(x_norm * (NX - 1));
            x = std::max(0, std::min(x, NX-1));
            
            int node_uy = center_y * NX + x;
            float uy_lbm = solver.h_u[2*node_uy + 1] * (1.0f/u_max); // scale for ghias u_max = 1.0f
            
            float diff_uy = uy_lbm - uy_ghia;
            error_sum_uy += diff_uy * diff_uy;
            ref_sum_uy += uy_ghia * uy_ghia;
            
            // printf("%-8.4f %-8d %-10.6f %-10.6f %-10.6f | %-8d %-10.6f %-10.6f %-10.6f\n", 
            //       y_norm, y, ux_lbm, ux_ghia, diff_ux,
            //       x, uy_lbm, uy_ghia, diff_uy);
        }
        
        float rmse_ux = (ref_sum_ux > 0.0f) ? sqrt(error_sum_ux / ref_sum_ux) : 0.0f;
        float rmse_uy = (ref_sum_uy > 0.0f) ? sqrt(error_sum_uy / ref_sum_uy) : 0.0f;
        
        float total_error = 100.0f * (rmse_ux + rmse_uy) / 2.0f;
        
        // printf("\nGhia Validation Summary (Re=%d):\n", re);
        // printf("  ux-velocity NRMSE: %.4f%%\n", rmse_ux * 100.0f);
        // printf("  uy-velocity NRMSE: %.4f%%\n", rmse_uy * 100.0f);
        // printf("  Combined    NRMSE: %.4f%%\n", total_error);
        
        return total_error;
    }
};

#endif // !LID_DRIVEN_SCENARIO_H