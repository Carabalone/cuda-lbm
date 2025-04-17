#ifndef POISEUILLE_SCENARIO_H
#define POISEUILLE_SCENARIO_H

#include "scenarios/scenario.cuh"
#include "poiseuilleFunctors.cuh"


struct PoiseuilleScenario : public ScenarioTrait <
    PoiseuilleInit,
    PoiseuilleBoundary,
    PoiseuilleValidation,
    BGK<2>
    // CM<NoAdapter>
> {
    static constexpr float u_max = 0.05f;
    static constexpr float viscosity = 1.0f/6.0f;
    static constexpr float tau = viscosity_to_tau(viscosity);
    static constexpr float omega = 1.0f / tau;

    static constexpr float S[quadratures] = {
        0.0f,    // density (conserved)
        omega,    // bulk viscosity (compressibility control)
        omega,    // energy flux tensor (stability control)
        0.0f,    // momentum-x (conserved)
        omega,    // energy square moment (stability for high Re)
        0.0f,    // momentum-y (conserved)
        omega,    // third-order moment (stability control)
        omega,   // shear viscosity (controls physical viscosity)
        omega    // shear viscosity (controls physical viscosity)
    };
    
    static const char* name() { return "Poiseuille"; }
    
    static InitType init() { 
        return InitType(u_max); 
    }

    static BoundaryType boundary() {
        return BoundaryType();
    }
    
    static ValidationType validation() {
        return ValidationType(u_max, viscosity);
    }

    static void add_bodies() {
        constexpr float D = 16.0f;               
        constexpr float r = D/2.0f;             
        constexpr float cx = 48.0f;              // ~3D from inlet
        constexpr float cy = NY/2.0f;            

        IBM_bodies.push_back(create_cylinder(cx, cy, r));
    }
    
    template <typename LBMSolver>
    static float compute_error(LBMSolver& solver) {
        auto analytical_profile = validation().getProfile();
        
        if (solver.update_ts < solver.timestep)
            solver.update_macroscopics();
        
        // compute just a slice
        float error_sum = 0.0f;
        for (int y = 0; y < NY; y++) {
            float sum_ux = 0.0f;
            for (int x = 0; x < NX; x++) {
                int index = (y * NX + x) * dimensions;
                sum_ux += solver.h_u[index];
            }
            float avg_ux = sum_ux / NX;
            
            float diff = avg_ux - analytical_profile[y];
            error_sum += diff * diff;
        }
        
        return std::sqrt(error_sum / NY) * 100.0f / u_max;
    }
};

#endif // !POISEUILLE_SCENARIO_H